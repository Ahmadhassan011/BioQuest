#!/usr/bin/env python3
"""
BioQuest Benchmark Suite.

Runs predictive, generative, optimization, ablation, and system benchmarks
and produces a JSON scorecard that is easily diffable across runs.

Usage:
    python scripts/benchmark.py              # full benchmark
    python scripts/benchmark.py --quick      # 1 trial, assays limited to 2
    python scripts/benchmark.py --output scorecard.json
    python scripts/benchmark.py --predictive-only
    python scripts/benchmark.py --n-trials 5
"""

import argparse
import json
import logging
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("benchmark")

SEED_MOLECULES = ["CCO", "c1ccccc1", "CC(=O)O", "CC", "C"]
N_TRIALS_DEFAULT = 3

TOX21_ASSAYS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _aggregate(trial_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Convert list of per-trial dicts to {metric: {mean, std, min, max, n}}."""
    if not trial_metrics:
        return {}
    keys = trial_metrics[0].keys()
    return {
        key: {
            "mean": float(np.mean([m[key] for m in trial_metrics])),
            "std": float(np.std([m[key] for m in trial_metrics])),
            "min": float(np.min([m[key] for m in trial_metrics])),
            "max": float(np.max([m[key] for m in trial_metrics])),
            "n": len(trial_metrics),
        }
        for key in keys
    }


def _predictive_benchmark(
    models_dir: str, use_gpu: bool, n_trials: int, quick: bool
) -> Dict[str, Any]:
    """Benchmark predictive models on held-out test sets."""
    from src.data.preparation.dti import DTIDatasetPreparer, DTIGraphDataset
    from src.data.preparation.toxicity import Tox21DatasetPreparer
    from src.data.preparation.property import PropertyDatasetPreparer, PropertyPredictionDataset
    from src.models.loader import ModelLoader
    from src.training.utils import create_data_loaders
    from src.training.gnn_dti import GNNDTITrainer
    from src.training.toxicity import ToxicityClassifierTrainer
    from src.training.property import PropertyPredictorTrainer
    import torch
    from torch.utils.data import TensorDataset

    results: Dict[str, List] = {}
    loader = ModelLoader(models_dir=models_dir, use_gpu=use_gpu)

    assays = TOX21_ASSAYS[:2] if quick else TOX21_ASSAYS

    for trial in range(n_trials):
        logger.info(f"Predictive benchmark trial {trial + 1}/{n_trials}")

        # --- DTI ---
        try:
            prep = DTIDatasetPreparer()
            data_list, splits, meta = prep.prepare_dti_dataset("DAVIS")
            dataset = DTIGraphDataset(data_list)
            _, _, test_loader = create_data_loaders(dataset, splits, dataset_type="dti")
            model = loader.load_dti_model()
            trainer = GNNDTITrainer(model, device=loader.device)
            dti_metrics = trainer.validate(test_loader)
            results.setdefault("dti_davis", []).append(
                {k: float(v) for k, v in dti_metrics.items()}
            )
        except Exception as e:
            logger.warning(f"DTI benchmark failed: {e}")

        # --- Toxicity (all assays) ---
        for assay in assays:
            try:
                prep = Tox21DatasetPreparer()
                X, y, splits, meta = prep.prepare_tox21_dataset(assay)
                dataset = TensorDataset(
                    torch.from_numpy(X).float(), torch.from_numpy(y).float()
                )
                _, _, test_loader = create_data_loaders(
                    dataset, splits, dataset_type="toxicity"
                )
                model = loader.load_toxicity_model()
                trainer = ToxicityClassifierTrainer(model, device=loader.device)
                tox_metrics = trainer.validate(test_loader)
                results.setdefault(f"tox21_{assay}", []).append(
                    {k: float(v) for k, v in tox_metrics.items()}
                )
            except Exception as e:
                logger.warning(f"Toxicity {assay} benchmark failed: {e}")

        # --- Property ---
        try:
            prep = PropertyDatasetPreparer()
            features, targets_dict, splits, meta = prep.prepare_property_dataset(
                "Lipophilicity_AstraZeneca"
            )
            dataset = PropertyPredictionDataset(features, targets_dict)
            _, _, test_loader = create_data_loaders(
                dataset, splits, dataset_type="property"
            )
            model = loader.load_property_model()
            trainer = PropertyPredictorTrainer(model, device=loader.device)
            prop_metrics = trainer.validate(test_loader)
            results.setdefault("property_lipophilicity", []).append(
                {k: float(v) for k, v in prop_metrics.items()}
            )
        except Exception as e:
            logger.warning(f"Property benchmark failed: {e}")

    return {task: _aggregate(metrics) for task, metrics in results.items()}


def _generative_benchmark(
    models_dir: str, use_gpu: bool, n_trials: int
) -> Dict[str, Any]:
    """Benchmark VAE generation quality."""
    from src.inference.vae import VAEGenerator
    from src.evaluation.generation import compute_all_generation_metrics

    results = []
    for trial in range(n_trials):
        logger.info(f"Generative benchmark trial {trial + 1}/{n_trials}")
        gen = VAEGenerator(models_dir=models_dir, use_gpu=use_gpu)
        molecules = gen.generate(500)
        if not molecules:
            logger.warning("VAE generated no molecules")
            continue
        metrics = compute_all_generation_metrics(molecules, reference_smiles=None)
        results.append(metrics)

    if not results:
        return {}

    return _aggregate(results)


def _optimization_benchmark(
    models_dir: str, use_gpu: bool, n_trials: int
) -> Dict[str, Any]:
    """Benchmark optimization success rate, convergence speed, Pareto front."""
    from src.inference import MoleculePredictor
    from src.core.optimization import OptimizationEvaluator
    from src.core.agents import GeneratorAgent, EvaluatorAgent, RefinerAgent, AgentOrchestrator

    objectives = {"affinity": 0.4, "toxicity": 0.3, "qed": 0.2, "sa": 0.1}
    protein = "MKFLK" * 50

    results = []
    for trial in range(n_trials):
        logger.info(f"Optimization benchmark trial {trial + 1}/{n_trials}")

        try:
            predictor = MoleculePredictor(
                protein, use_gpu=use_gpu, models_dir=models_dir
            )
            opt = OptimizationEvaluator(objective_weights=objectives, patience=10)
            vae = predictor.get_vae_generator()

            orchestrator = AgentOrchestrator(
                GeneratorAgent(vae),
                EvaluatorAgent(predictor, opt),
                RefinerAgent(opt),
            )

            start = time.time()
            for it in range(1, 21):
                term, _ = orchestrator.run_iteration(
                    SEED_MOLECULES,
                    objectives,
                    iteration=it,
                    max_iterations=20,
                    batch_size=50,
                )
                if not term:
                    break
            elapsed = time.time() - start

            best = opt.get_best_molecule()
            pareto = opt.get_pareto_front()
            conv = opt.get_convergence_metrics()

            results.append({
                "success": 1.0 if best and best.composite_score > 0.6 else 0.0,
                "best_score": float(best.composite_score) if best else 0.0,
                "pareto_size": len(pareto),
                "convergence_iterations": conv.get("statistics", {}).get(
                    "current_iteration", 0
                ),
                "execution_time_s": elapsed,
            })
        except Exception as e:
            logger.warning(f"Optimization trial failed: {e}")

    if not results:
        return {}

    return _aggregate(results)


def _ablation_benchmark(
    models_dir: str, use_gpu: bool, n_trials: int
) -> Dict[str, Any]:
    """Run optimization in each ablation mode and compare."""
    from src.inference import MoleculePredictor
    from src.core.optimization import OptimizationEvaluator
    from src.core.agents import GeneratorAgent, EvaluatorAgent, RefinerAgent, AgentOrchestrator

    objectives = {"affinity": 0.4, "toxicity": 0.3, "qed": 0.2, "sa": 0.1}
    protein = "MKFLK" * 50
    modes = ["full", "no_refiner", "no_generator", "single_pass"]

    ablation_results: Dict[str, List] = {}
    for mode in modes:
        for trial in range(n_trials):
            logger.info(f"Ablation trial {trial + 1}/{n_trials} — mode={mode}")
            try:
                predictor = MoleculePredictor(
                    protein, use_gpu=use_gpu, models_dir=models_dir
                )
                opt = OptimizationEvaluator(objective_weights=objectives, patience=10)
                vae = predictor.get_vae_generator()

                orchestrator = AgentOrchestrator(
                    GeneratorAgent(vae),
                    EvaluatorAgent(predictor, opt),
                    RefinerAgent(opt),
                    ablation_mode=mode,
                )
                for it in range(1, 21):
                    term, _ = orchestrator.run_iteration(
                        SEED_MOLECULES,
                        objectives,
                        iteration=it,
                        max_iterations=20,
                        batch_size=50,
                    )
                    if not term:
                        break
                best = opt.get_best_molecule()
                ablation_results.setdefault(mode, []).append(
                    float(best.composite_score) if best else 0.0
                )
            except Exception as e:
                logger.warning(f"Ablation mode={mode} trial {trial} failed: {e}")
                ablation_results.setdefault(mode, []).append(0.0)

    return {
        mode: {
            "mean_best_score": float(np.mean(scores)),
            "std_best_score": float(np.std(scores)),
            "n_trials": len(scores),
        }
        for mode, scores in ablation_results.items()
    }


def _system_benchmark(
    models_dir: str, use_gpu: bool, n_trials: int
) -> Dict[str, Any]:
    """Benchmark inference latency with warm-up rounds."""
    from src.inference import MoleculePredictor

    latencies = []
    try:
        predictor = MoleculePredictor(
            "MKFLK" * 50, use_gpu=use_gpu, models_dir=models_dir
        )
        test_smiles = (
            ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CO", "CCCl", "CCCC"] * 20
        )

        # Warm-up: run an unseen batch to cold-start any lazy init / JIT compilation
        logger.info("System benchmark warm-up round...")
        predictor.batch_predict(test_smiles[:50])
        predictor.batch_predict(test_smiles)

        # Measured rounds
        for i in range(n_trials):
            logger.info(f"System benchmark trial {i + 1}/{n_trials}")
            start = time.time()
            predictor.batch_predict(test_smiles)
            elapsed = time.time() - start
            latencies.append(elapsed)

        mean_lat = float(np.mean(latencies))
        std_lat = float(np.std(latencies))
        n_mols = len(test_smiles)

        return {
            "inference_batch_time_s": {
                "mean": mean_lat,
                "std": std_lat,
                "min": float(np.min(latencies)),
                "max": float(np.max(latencies)),
                "n": n_trials,
            },
            "molecules_per_second": {
                "mean": float(n_mols / mean_lat),
                "std": float(
                    abs(n_mols / mean_lat - n_mols / (mean_lat + std_lat))
                ),
            },
            "batch_size": n_mols,
            "n_warmup_rounds": 2,
        }
    except Exception as e:
        logger.warning(f"System benchmark failed: {e}")

    return {}


def _validate_scorecard(scorecard: Dict) -> None:
    """Validate benchmark scorecard structure before serialisation."""
    required_top = {"benchmark_version", "timestamp", "git_hash", "config",
                    "predictive", "generative", "optimization", "ablation", "system"}
    missing = required_top - set(scorecard.keys())
    if missing:
        raise ValueError(f"Scorecard missing required keys: {missing}")

    cfg = scorecard.get("config", {})
    if not isinstance(cfg.get("n_trials"), int):
        raise TypeError(f"config.n_trials must be int, got {type(cfg.get('n_trials'))}")
    if not isinstance(cfg.get("quick"), bool):
        raise TypeError(f"config.quick must be bool, got {type(cfg.get('quick'))}")

    for section in ("predictive", "generative", "optimization", "ablation", "system"):
        if not isinstance(scorecard.get(section), dict):
            raise TypeError(f"{section} must be a dict, got {type(scorecard.get(section))}")

    for section in ("predictive", "generative", "optimization"):
        for task, metrics in scorecard.get(section, {}).items():
            if not isinstance(metrics, dict):
                continue
            # Flat aggregated metric: {"mean": 0.5, "std": 0.01, ...}
            if "mean" in metrics:
                if not isinstance(metrics["mean"], (int, float)):
                    raise TypeError(
                        f"{section}.{task}.mean must be numeric, "
                        f"got {type(metrics['mean'])}"
                    )
                continue
            # Nested task container: {"accuracy": {"mean": 0.5, ...}, ...}
            for sub_name, sub_metrics in metrics.items():
                if isinstance(sub_metrics, dict) and "mean" in sub_metrics:
                    if not isinstance(sub_metrics["mean"], (int, float)):
                        raise TypeError(
                            f"{section}.{task}.{sub_name}.mean must be numeric, "
                            f"got {type(sub_metrics['mean'])}"
                        )


def _generate_evaluation_reports(scorecard: Dict, output_dir: str = "artifacts/reports") -> None:
    """Generate per-section JSON reports using EvaluationReporter."""
    from src.evaluation.reporter import EvaluationReporter

    reporter = EvaluationReporter(output_dir=output_dir)
    model_name = f"BioQuest_{scorecard.get('git_hash', 'unknown')}"

    for section in ("predictive", "generative", "optimization", "ablation", "system"):
        data = scorecard.get(section, {})
        if not data:
            continue
        if section in ("predictive",):
            for task, metrics in data.items():
                if isinstance(metrics, dict):
                    reporter.generate_json_report(metrics, model_name, task)
        elif section == "generative":
            reporter.generate_json_report(data, model_name, section)
        else:
            for mode, metrics in data.items():
                if isinstance(metrics, dict):
                    reporter.generate_json_report(
                        metrics, model_name, f"{section}_{mode}"
                    )

    reporter.print_report(scorecard.get("predictive", {}), model_name)


def _scorecard(
    predictive: Dict,
    generative: Dict,
    optimization: Dict,
    ablation: Dict,
    system: Dict,
    baselines: Dict,
    statistical: Dict,
    n_trials: int,
    quick: bool,
) -> Dict:
    """Assemble everything into a diffable scorecard."""
    scorecard = {
        "benchmark_version": "1.1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "config": {
            "n_trials": n_trials,
            "quick": quick,
            "baselines_run": bool(baselines.get("generative") or baselines.get("predictive")),
        },
        "predictive": predictive,
        "generative": generative,
        "optimization": optimization,
        "ablation": ablation,
        "system": system,
    }

    if baselines:
        scorecard["baselines"] = baselines
    if statistical:
        scorecard["statistical_comparison"] = statistical

    return scorecard


def _statistical_comparison(
    bioquest_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """Compare BioQuest vs baseline metrics using a direction-based heuristic.

    Computes the raw difference (BioQuest − baseline) per metric and applies
    a higher-is-better sign check. Returns effect direction only — no
    statistical testing is performed.
    """

    comparison = {}
    shared_keys = set(bioquest_metrics.keys()) & set(baseline_metrics.keys())
    for key in sorted(shared_keys):
        bq_val = bioquest_metrics.get(key, 0.0)
        bl_val = baseline_metrics.get(key, 0.0)

        higher_is_better = key not in ("rmse", "mae", "loss")
        diff = bq_val - bl_val
        bioquest_better = (diff > 0) if higher_is_better else (diff < 0)

        comparison[key] = {
            "bioquest": bq_val,
            "baseline": bl_val,
            "difference": round(diff, 6),
            "bioquest_better": bool(bioquest_better),
        }

    return comparison


def _baseline_benchmark(models_dir: str) -> Dict[str, Any]:
    """Run baseline models and return their results."""
    logger.info("=" * 60)
    logger.info("BASELINE BENCHMARKS")
    logger.info("=" * 60)

    results: Dict[str, Any] = {"generative": {}, "predictive": {}}

    try:
        from src.models.baselines.reinvent_wrapper import run_reinvent_generation
        gen_result = run_reinvent_generation(num_molecules=500)
        results["generative"] = {
            "metrics": gen_result.get("metrics", {}),
            "surrogate": gen_result.get("surrogate", True),
        }
        logger.info(f"REINVENT baseline done (surrogate={gen_result.get('surrogate')})")
    except Exception as e:
        logger.warning(f"REINVENT baseline failed: {e}")

    try:
        from src.models.baselines.deeppurpose_wrapper import run_deeppurpose_predictive
        pred_result = run_deeppurpose_predictive()
        results["predictive"] = {
            "tasks": pred_result.get("tasks", {}),
            "model": pred_result.get("model", "deeppurpose"),
        }
        logger.info("DeepPurpose baseline done")
    except Exception as e:
        logger.warning(f"DeepPurpose baseline failed: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="BioQuest Benchmark Suite")
    parser.add_argument("--models-dir", default="artifacts/models")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--quick", action="store_true", help="1 trial, 2 assays")
    parser.add_argument(
        "--output", default="artifacts/benchmark_scorecard.json"
    )
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    parser.add_argument("--predictive-only", action="store_true")
    parser.add_argument("--generative-only", action="store_true")
    parser.add_argument("--optimization-only", action="store_true")
    parser.add_argument("--ablation-only", action="store_true")
    parser.add_argument("--system-only", action="store_true")
    parser.add_argument("--run-baselines", action="store_true",
                        help="Run external baseline model comparison")
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 1

    run_all = not (
        args.predictive_only
        or args.generative_only
        or args.optimization_only
        or args.ablation_only
        or args.system_only
    )

    predictive: Dict = {}
    generative: Dict = {}
    optimization: Dict = {}
    ablation: Dict = {}
    system: Dict = {}
    baselines: Dict = {}
    statistical: Dict = {}

    if args.run_baselines:
        baselines = _baseline_benchmark(args.models_dir)

    if run_all or args.predictive_only:
        logger.info("=" * 60)
        logger.info("PREDICTIVE BENCHMARKS")
        logger.info("=" * 60)
        predictive = _predictive_benchmark(
            args.models_dir, args.use_gpu, args.n_trials, args.quick
        )
        logger.info(f"Done: {len(predictive)} datasets")

    if run_all or args.generative_only:
        logger.info("=" * 60)
        logger.info("GENERATIVE BENCHMARKS")
        logger.info("=" * 60)
        generative = _generative_benchmark(
            args.models_dir, args.use_gpu, args.n_trials
        )
        logger.info(f"Done: {len(generative)} metrics")

    if run_all or args.optimization_only:
        logger.info("=" * 60)
        logger.info("OPTIMIZATION BENCHMARKS")
        logger.info("=" * 60)
        optimization = _optimization_benchmark(
            args.models_dir, args.use_gpu, args.n_trials
        )
        logger.info(f"Done: {len(optimization)} metrics")

    if run_all or args.ablation_only:
        logger.info("=" * 60)
        logger.info("ABLATION STUDIES")
        logger.info("=" * 60)
        ablation = _ablation_benchmark(
            args.models_dir, args.use_gpu, args.n_trials
        )
        logger.info(f"Done: {len(ablation)} modes")

    if run_all or args.system_only:
        logger.info("=" * 60)
        logger.info("SYSTEM BENCHMARKS")
        logger.info("=" * 60)
        system = _system_benchmark(
            args.models_dir, args.use_gpu, args.n_trials
        )
        logger.info(f"Done: {len(system)} metrics")

    # Statistical comparison between BioQuest and baselines
    if baselines:
        logger.info("=" * 60)
        logger.info("STATISTICAL COMPARISON")
        logger.info("=" * 60)

        # Compare generative metrics
        gen_baseline = baselines.get("generative", {}).get("metrics", {})
        if generative and gen_baseline:
            stat_gen = _statistical_comparison(
                {k: v.get("mean", 0) for k, v in generative.items()
                 if isinstance(v, dict) and "mean" in v},
                gen_baseline,
            )
            if stat_gen:
                statistical["generative"] = stat_gen

        # Compare predictive metrics per task
        pred_baselines = baselines.get("predictive", {}).get("tasks", {})
        if predictive and pred_baselines:
            stat_pred = {}
            for task in pred_baselines:
                bq_metrics = predictive.get(task, {})
                bl_metrics = pred_baselines.get(task, {})
                if isinstance(bq_metrics, dict) and "mean" in bq_metrics:
                    flat_bq = {k: v.get("mean", 0) for k, v in bq_metrics.items()
                               if isinstance(v, dict) and "mean" in v}
                    flat_bl = bl_metrics
                    task_comp = _statistical_comparison(flat_bq, flat_bl)
                    if task_comp:
                        stat_pred[task] = task_comp
            if stat_pred:
                statistical["predictive"] = stat_pred

        logger.info(f"Compared {len(statistical)} groups")

    scorecard = _scorecard(
        predictive, generative, optimization, ablation, system,
        baselines, statistical,
        args.n_trials, args.quick,
    )

    _validate_scorecard(scorecard)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scorecard, f, indent=2, sort_keys=False)
    logger.info(f"Benchmark scorecard saved to {output_path}")

    _generate_evaluation_reports(scorecard)

    print("\n" + "=" * 60)
    print("BIOQUEST BENCHMARK SCORECARD")
    print("=" * 60)
    ds = len(predictive)
    print(f"Predictive datasets: {ds}")
    gen_keys = list(generative.keys())
    if gen_keys:
        v = generative.get("validity", generative.get(gen_keys[0], {}))
        print(f"Validity: {v.get('mean', 0):.3f} ± {v.get('std', 0):.3f}")
    print(f"Ablation modes: {len(ablation)}")
    if system:
        lat = system.get("inference_batch_time_s", {})
        print(f"Inference latency: {lat.get('mean', 0):.3f}s ± {lat.get('std', 0):.3f}s")
    if baselines:
        print("Baselines: REINVENT + DeepPurpose")
    print("=" * 60)


if __name__ == "__main__":
    main()
