"""
CLI entry point for BioQuest.

Command-line interface for running BioQuest optimization.
"""

import json
import logging
import sys
import time
import argparse
from pathlib import Path

from src.core.agents import AgentOrchestrator, GeneratorAgent, EvaluatorAgent, RefinerAgent
from src.core.optimization import OptimizationEvaluator
from src.inference import MoleculePredictor, ModelNotLoadedError
from src.core.types import validate_protein_sequence

logger = logging.getLogger(__name__)


def run_optimization(
    protein_sequence: str,
    seeds: list,
    objectives: dict,
    max_iterations: int = 50,
    batch_size: int = 50,
    output: str = "bioquest_results.json",
) -> dict:
    """
    Run molecule optimization.

    Args:
        protein_sequence: Target protein sequence
        seeds: List of seed SMILES
        objectives: Dictionary of objective weights
        max_iterations: Maximum optimization iterations
        batch_size: Batch size for evaluation
        output: Output file path

    Returns:
        Results dictionary
    """
    logger.info(f"Starting optimization for protein: {protein_sequence[:20]}...")

    try:
        validated_seq = validate_protein_sequence(protein_sequence)
    except ValueError as e:
        logger.error(f"Invalid protein sequence: {e}")
        return {"error": str(e)}

    try:
        predictor = MoleculePredictor(
            protein_sequence=validated_seq,
            use_gpu=False,
            models_dir="artifacts/models",
        )
    except ModelNotLoadedError as e:
        logger.error(f"Failed to load models: {e}")
        return {"error": str(e)}

    opt_evaluator = OptimizationEvaluator(
        objective_weights=objectives,
        plateau_threshold=0.001,
        patience=20,
    )

    vae_generator = predictor.get_vae_generator()
    if vae_generator is None:
        logger.error("VAE generator not available")
        return {"error": "VAE generator not available"}

    generator = GeneratorAgent(vae_generator)
    evaluator_agent = EvaluatorAgent(predictor, opt_evaluator)
    refiner = RefinerAgent(opt_evaluator)

    orchestrator = AgentOrchestrator(
        generator_agent=generator,
        evaluator_agent=evaluator_agent,
        refiner_agent=refiner,
    )

    start_time = time.time()

    for iteration in range(1, max_iterations + 1):
        logger.info(f"Iteration {iteration}/{max_iterations}")

        terminate, reason = orchestrator.run_iteration(
            seeds=seeds,
            objectives=objectives,
            iteration=iteration,
            max_iterations=max_iterations,
            batch_size=batch_size,
        )

        if terminate:
            logger.info(f"Termination: {reason}")
            break

    elapsed = time.time() - start_time

    best = evaluator_agent.get_best_molecule()
    top_5 = evaluator_agent.get_top_molecules(k=5)
    pareto = evaluator_agent.get_pareto_front()

    results = {
        "protein_sequence": protein_sequence,
        "objectives": objectives,
        "total_iterations": iteration,
        "execution_time_seconds": elapsed,
        "best_molecule": best.to_dict() if best else None,
        "top_5": [m.to_dict() for m in top_5],
        "pareto_front": [m.to_dict() for m in pareto],
    }

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output}")
    return results


def main(args=None):
    """
    Main CLI entry point.

    Args:
        args: Optional command-line arguments

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="BioQuest - AI Drug Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (JSON)",
    )

    parser.add_argument(
        "--protein",
        type=str,
        default=None,
        help="Target protein sequence",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        nargs="+",
        default=None,
        help="Seed molecule SMILES",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Maximum optimization iterations",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for evaluation",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="bioquest_results.json",
        help="Output results file",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parsed_args = parser.parse_args(args)

    level = logging.DEBUG if parsed_args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if parsed_args.config:
        config_path = Path(parsed_args.config)
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                protein_seq = config.get("protein_sequence", parsed_args.protein)
                seeds = config.get("seeds", parsed_args.seeds or [])
                objectives = config.get("objectives", {"affinity": 0.4, "toxicity": 0.3, "qed": 0.2, "sa": 0.1})
                max_iterations = config.get("max_iterations", parsed_args.iterations)
                batch_size = config.get("batch_size", parsed_args.batch_size)
        else:
            logger.error(f"Config file not found: {config_path}")
            return 1
    else:
        if not parsed_args.protein:
            logger.error("--protein required when not using --config")
            return 1

        protein_seq = parsed_args.protein
        seeds = parsed_args.seeds or []
        objectives = {"affinity": 0.4, "toxicity": 0.3, "qed": 0.2, "sa": 0.1}
        max_iterations = parsed_args.iterations
        batch_size = parsed_args.batch_size

    if not seeds:
        seeds = ["CCO", "c1ccccc1", "CC(=O)O", "CC", "C"]

    logger.info("BioQuest CLI initialized")
    logger.info(f"Protein: {protein_seq[:30]}...")
    logger.info(f"Seeds: {len(seeds)} molecules")
    logger.info(f"Max iterations: {max_iterations}")

    try:
        results = run_optimization(
            protein_sequence=protein_seq,
            seeds=seeds,
            objectives=objectives,
            max_iterations=max_iterations,
            batch_size=batch_size,
            output=parsed_args.output,
        )

        if "error" in results:
            logger.error(f"Optimization failed: {results['error']}")
            return 1

        logger.info(f"Optimization complete in {results['execution_time_seconds']:.1f}s")
        best = results.get("best_molecule")
        if best:
            logger.info(f"Best molecule: {best['smiles']}")
            logger.info(f"Best score: {best['composite_score']:.4f}")

        return 0

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())