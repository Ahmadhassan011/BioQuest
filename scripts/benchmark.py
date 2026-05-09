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
            ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CO", "CCCl", "CCCC"] * 150
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


def _scorecard(
    predictive: Dict,
    generative: Dict,
    optimization: Dict,
    ablation: Dict,
    system: Dict,
    n_trials: int,
    quick: bool,
) -> Dict:
    """Assemble everything into a diffable scorecard."""
    scorecard = {
        "benchmark_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "config": {
            "n_trials": n_trials,
            "quick": quick,
        },
        "predictive": predictive,
        "generative": generative,
        "optimization": optimization,
        "ablation": ablation,
        "system": system,
    }

    return scorecard


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

    scorecard = _scorecard(
        predictive, generative, optimization, ablation, system,
        args.n_trials, args.quick,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scorecard, f, indent=2, sort_keys=False)
    logger.info(f"Benchmark scorecard saved to {output_path}")

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
    print("=" * 60)


if __name__ == "__main__":
    main()
