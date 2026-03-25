"""
Main Module: BioQuest orchestration, initialization, and execution.

Entry point for the complete BioQuest pipeline.
Coordinates all components:
- Data loading and validation
- Model initialization
- Agent orchestration
- Optimization loop with termination logic
- Results collection and reporting
"""

import logging
import sys
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import argparse
import subprocess
from src.data import create_dataset
from src.pipelines.prediction import MoleculePredictor
from src.pipelines.generation import HybridMoleculeGenerator
from src.pipelines.optimization import OptimizationEvaluator
from src.app.core import (
    GeneratorAgent,
    EvaluatorAgent,
    RefinerAgent,
    AgentOrchestrator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bioquest.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging level and format.

    Args:
        verbose: If True, set to DEBUG level
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)
    logger.info(f"Logging configured at {level} level")


def validate_configuration(config: Dict[str, Any]) -> bool:
    """
    Validate input configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    logger.info("Validating configuration...")

    # Check protein sequence
    if not config.get("protein_sequence"):
        logger.error("Protein sequence not provided")
        return False

    if len(config["protein_sequence"]) < 5:
        logger.error("Protein sequence too short (minimum 5 amino acids)")
        return False

    # Check seeds
    if not config.get("seeds") or len(config["seeds"]) < 1:
        logger.error("At least one seed molecule required")
        return False

    # Check objectives
    objectives = config.get("objectives", {})
    if not objectives:
        logger.error("No optimization objectives specified")
        return False

    total_weight = sum(objectives.values())
    if total_weight <= 0:
        logger.error("Total objective weight must be positive")
        return False

    # Check iterations
    max_iterations = config.get("max_iterations", 50)
    if max_iterations < 1 or max_iterations > 500:
        logger.error("max_iterations must be between 1 and 500")
        return False

    logger.info("✓ Configuration validated successfully")
    return True


def initialize_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize all BioQuest components.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of initialized components
    """
    logger.info("\n" + "=" * 70)
    logger.info("INITIALIZING BioQuest COMPONENTS")
    logger.info("=" * 70 + "\n")

    try:
        # Step 1: Create dataset
        logger.info("Step 1/5: Creating dataset...")
        dataset_result = create_dataset(
            protein_sequence=config["protein_sequence"],
            seed_smiles=config["seeds"],
            objectives=config["objectives"],
        )

        # Unpack dataset result (returns tuple: data_list, splits, metadata)
        if isinstance(dataset_result, tuple):
            data_list, splits, metadata = dataset_result
            train_size = (
                len(splits.get("train", []))
                if isinstance(splits.get("train"), list)
                else splits.get("train", 0)
            )
            val_size = (
                len(splits.get("val", []))
                if isinstance(splits.get("val"), list)
                else splits.get("val", 0)
            )
            test_size = (
                len(splits.get("test", []))
                if isinstance(splits.get("test"), list)
                else splits.get("test", 0)
            )
            logger.info(
                f"✓ Dataset created: {len(data_list)} samples (train={train_size}, val={val_size}, test={test_size})"
            )
            dataset = type(
                "Dataset",
                (),
                {
                    "get_seeds": lambda self: config["seeds"],
                    "get_objectives": lambda self: config["objectives"],
                },
            )()
        else:
            dataset = dataset_result
            logger.info("Dataset created")

        # Step 2: Initialize predictor
        logger.info("Step 2/5: Initializing molecular predictor...")
        predictor = MoleculePredictor(
            protein_sequence=config["protein_sequence"],
            use_gpu=config.get("use_gpu", False),
            models_dir=config.get("models_dir", "trained_models"),
        )
        logger.info(
            "✓ Predictor initialized (using custom trained models with heuristic fallback)"
        )

        # Step 3: Initialize generator
        logger.info("Step 3/5: Initializing hybrid molecule generator...")
        generator = HybridMoleculeGenerator(
            vae_enabled=config.get("vae_enabled", True),
            evolutionary_enabled=config.get("evolutionary_enabled", True),
            device="cuda" if config.get("use_gpu", False) else "cpu",
        )
        logger.info("✓ Generator initialized (RDKit + VAE hybrid)")

        # Step 4: Initialize evaluator
        logger.info("Step 4/5: Initializing evaluator...")
        evaluator = OptimizationEvaluator(
            objective_weights=config["objectives"],
            plateau_threshold=config.get("plateau_threshold", 0.001),
            patience=config.get("patience", 20),
        )
        logger.info("✓ Evaluator initialized (multi-objective + convergence)")

        # Step 5: Initialize agents
        logger.info("Step 5/5: Initializing agent orchestrator...")
        gen_agent = GeneratorAgent(generator)
        eval_agent = EvaluatorAgent(predictor, evaluator)
        ref_agent = RefinerAgent(evaluator)
        orchestrator = AgentOrchestrator(gen_agent, eval_agent, ref_agent)
        logger.info("✓ Agents initialized (Generator, Evaluator, Refiner)")

        logger.info("\n" + "=" * 70)
        logger.info("✓ ALL COMPONENTS INITIALIZED SUCCESSFULLY")
        logger.info("=" * 70 + "\n")

        return {
            "dataset": dataset,
            "predictor": predictor,
            "generator": generator,
            "evaluator": evaluator,
            "orchestrator": orchestrator,
            "gen_agent": gen_agent,
            "eval_agent": eval_agent,
            "ref_agent": ref_agent,
        }

    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        raise


def run_optimization_loop(
    components: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run main optimization loop.

    Args:
        components: Dictionary of initialized components
        config: Configuration dictionary

    Returns:
        Final results dictionary
    """
    logger.info("\n" + "=" * 70)
    logger.info("STARTING OPTIMIZATION LOOP")
    logger.info("=" * 70 + "\n")

    orchestrator = components["orchestrator"]
    dataset = components["dataset"]
    seeds = dataset.get_seeds()
    objectives = dataset.get_objectives()
    max_iterations = config.get("max_iterations", 50)
    batch_size = config.get("batch_size", 50)

    start_time = datetime.now()

    # Main optimization loop
    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        try:
            # Run single iteration with all agents
            should_continue, reason = orchestrator.run_iteration(
                seeds=seeds,
                objectives=objectives,
                iteration=iteration,
                max_iterations=max_iterations,
                batch_size=batch_size,
            )

            if not should_continue:
                logger.info(f"\nOptimization terminated: {reason}")
                break

        except KeyboardInterrupt:
            logger.warning("\n\nOptimization interrupted by user")
            break
        except Exception as e:
            logger.error(f"Iteration {iteration} failed: {e}", exc_info=True)
            if iteration >= max_iterations - 1:
                logger.warning("Continuing to final iteration...")
            else:
                logger.warning("Attempting to continue...")

    elapsed_time = datetime.now() - start_time

    # Collect final results
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION COMPLETE - COLLECTING RESULTS")
    logger.info("=" * 70 + "\n")

    results = orchestrator.get_final_results()
    results["execution_time_seconds"] = elapsed_time.total_seconds()
    results["execution_time_formatted"] = str(elapsed_time)

    return results


def print_results_summary(results: Dict[str, Any]) -> None:
    """
    Print summary of optimization results.

    Args:
        results: Results dictionary from optimization
    """
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION RESULTS SUMMARY")
    logger.info("=" * 70 + "\n")

    # Best molecule
    best = results.get("best_molecule")
    if best:
        logger.info("🏆 BEST MOLECULE:")
        logger.info(f"   SMILES: {best['smiles']}")
        logger.info(f"   Affinity: {best['affinity']:.4f}")
        logger.info(f"   Toxicity: {best['toxicity']:.4f} (lower is better)")
        logger.info(f"   QED: {best['qed']:.4f}")
        logger.info(f"   SA: {best['sa']:.4f}")
        logger.info(f"   Composite Score: {best['composite_score']:.4f}")

    # Statistics
    stats = results.get("agent_statistics", {})
    logger.info("\nSTATISTICS:")
    logger.info(f"   Total Iterations: {results['total_iterations']}")
    logger.info(f"   Total Molecules Generated: {results['total_molecules_generated']}")
    logger.info(f"   Total Molecules Evaluated: {results['total_molecules_evaluated']}")
    logger.info(f"   Execution Time: {results.get('execution_time_formatted')}")

    # Top 5 molecules
    top_5 = results.get("top_10", [])[:5]
    if top_5:
        logger.info("\n🥇 TOP 5 MOLECULES:")
        for i, mol in enumerate(top_5, 1):
            logger.info(
                f"   #{i}: {mol['smiles'][:50]}... (Score: {mol['composite_score']:.4f})"
            )

    # Convergence metrics
    convergence = results.get("convergence_metrics", {})
    conv_stats = convergence.get("convergence", {})
    if conv_stats:
        logger.info("\n📈 CONVERGENCE METRICS:")
        logger.info(f"   Best Score: {conv_stats['best_score']:.4f}")
        logger.info(f"   Best Iteration: {conv_stats['best_iteration']}")
        logger.info(
            f"   Iterations Since Improvement: {conv_stats['iterations_since_improvement']}"
        )

    # Pareto front
    pareto = results.get("pareto_front", [])
    logger.info(f"\n⚖️ PARETO FRONT: {len(pareto)} molecules")

    logger.info("\n" + "=" * 70 + "\n")


def save_results(results: Dict[str, Any], output_file: str) -> None:
    """
    Save results to JSON file.

    Args:
        results: Results dictionary
        output_file: Output filepath
    """
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"✓ Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def print_ethical_notice() -> None:
    """Print ethical considerations notice."""
    logger.info("""
    
╔════════════════════════════════════════════════════════════════════════════════╗
║                         ETHICAL CONSIDERATIONS                                 ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ This BioQuest (BioQuest) is designed for research                              ║
║ purposes only. Users and researchers must understand and acknowledge:          ║
║                                                                                ║
║ 1. VALIDATION REQUIREMENT:                                                     ║
║    - Generated molecules MUST undergo rigorous experimental validation         ║
║    - In silico predictions are approximations and require wet-lab testing      ║
║    - No molecule should be synthesized or tested without proper review         ║
║                                                                                ║
║ 2. RESPONSIBLE CONDUCT:                                                        ║
║    - Use results only for legitimate drug discovery research                   ║
║    - Follow institutional ethics boards and regulatory guidelines              ║
║    - Maintain transparency in methodology and limitations                      ║
║                                                                                ║
║ 3. LIMITATIONS:                                                                ║
║    - Model predictions are probabilistic and may have high error rates         ║
║    - Off-target effects and ADME properties require additional testing         ║
║    - This tool does not predict clinical efficacy or safety profiles           ║
║                                                                                ║
║ 4. DUAL-USE CONSIDERATIONS:                                                    ║
║    - This system must not be used for synthesis of harmful compounds           ║
║    - Report misuse to appropriate authorities immediately                      ║
║                                                                                ║
║ By using BioQuest, you agree to use this technology responsibly and ethically. ║
║ The authors assume no liability for misuse or harm resulting from this tool.   ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
    
    """)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main BioQuest entry point.

    Args:
        args: Command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="BioQuest - AI Drug Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file",
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
        help="Seed molecule SMILES strings",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Maximum optimization iterations (default: 50)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="bioquest_results.json",
        help="Output results file (default: bioquest_results.json)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch Streamlit UI instead of CLI",
    )

    parsed_args = parser.parse_args(args)

    # Setup logging
    setup_logging(parsed_args.verbose)

    # Print header
    logger.info("\n" + "=" * 70)
    logger.info("BioQuest v1.0.0")
    logger.info("AI-Driven Drug Discovery Pipeline")
    logger.info("=" * 70 + "\n")

    print_ethical_notice()

    try:
        # Load configuration
        config_path = parsed_args.config or "configs/config_default.json"
        logger.info(f"Loading configuration from {config_path}...")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Override with command-line arguments if provided
        if parsed_args.protein:
            config["protein_sequence"] = parsed_args.protein
        if parsed_args.seeds:
            config["seeds"] = parsed_args.seeds
        if parsed_args.iterations:
            config["max_iterations"] = parsed_args.iterations

        # Launch UI if requested
        if parsed_args.ui:
            logger.info("Launching Streamlit UI...")
            try:
                subprocess.run(["streamlit", "run", "src/app/ui.py"], check=True)
            except FileNotFoundError:
                logger.error(
                    "Streamlit not found. Please ensure it is installed "
                    "(`pip install streamlit`)"
                )
                return 1
            except Exception as e:
                logger.error(f"Failed to launch Streamlit UI: {e}")
                return 1
            return 0

        # Validate configuration
        if not validate_configuration(config):
            logger.error("Configuration validation failed")
            return 1

        # Initialize components
        components = initialize_components(config)

        # Run optimization
        results = run_optimization_loop(components, config)

        # Print summary
        print_results_summary(results)

        # Save results
        save_results(results, parsed_args.output)

        logger.info("BioQuest execution completed successfully\n")
        return 0

    except KeyboardInterrupt:
        logger.warning("\n\nExecution interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\n\nFATAL ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
