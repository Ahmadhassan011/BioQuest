"""
Agent Orchestrator: Coordinates agent workflow.

This orchestrator manages the interaction between Generator, Evaluator,
and Refiner agents via message passing.
"""

import logging
from typing import Dict, List, Tuple, Any

from .messages import AgentMessage
from .generator import GeneratorAgent
from .evaluator import EvaluatorAgent
from .refiner import RefinerAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Orchestrates agent workflow using message passing.

    Controls the interaction between Generator, Evaluator, and Refiner agents.
    """

    def __init__(
        self,
        generator_agent: GeneratorAgent,
        evaluator_agent: EvaluatorAgent,
        refiner_agent: RefinerAgent,
        ablation_mode: str = "full",
    ):
        """
        Initialize agent orchestrator.

        Args:
            generator_agent: GeneratorAgent instance
            evaluator_agent: EvaluatorAgent instance
            refiner_agent: RefinerAgent instance
            ablation_mode: One of "full", "no_refiner", "no_generator", "single_pass".
                "full" — all three agents.
                "no_refiner" — skip RefinerAgent (no strategy switching, no termination check).
                "no_generator" — use VAE only, skip evolutionary generation.
                "single_pass" — run one generation + evaluation pass, no iterative loop.
        """
        self.generator = generator_agent
        self.evaluator = evaluator_agent
        self.refiner = refiner_agent
        self.ablation_mode = ablation_mode
        self.name = "AgentOrchestrator"

        self.message_log: List[AgentMessage] = []
        self.iteration_count = 0

        logger.info(f"AgentOrchestrator initialized (mode={ablation_mode})")

    def run_iteration(
        self,
        seeds: List[str],
        objectives: Dict[str, float],
        iteration: int,
        max_iterations: int = 100,
        batch_size: int = 50,
    ) -> Tuple[bool, str]:
        """
        Run single optimization iteration with all agents.

        Args:
            seeds: Seed molecules
            objectives: Optimization objectives and weights
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations
            batch_size: Number of molecules per iteration

        Returns:
            Tuple of (should_continue, reason)
        """
        self.iteration_count = iteration

        logger.info(f"\n{'=' * 60}")
        logger.info(f"ITERATION {iteration}/{max_iterations}")
        logger.info(f"{'=' * 60}\n")

        if self.ablation_mode == "single_pass" and iteration > 1:
            return False, "Single pass ablation — one iteration only"

        try:
            # Step 1: Generator Agent
            if self.ablation_mode == "no_generator":
                molecules = list(seeds)
                logger.info(f"Step 1 skipped (ablation): using {len(molecules)} seeds as generation")
            else:
                logger.info("Step 1: Generator Agent - Creating new molecules...")
                strategy = "hybrid" if iteration == 0 else self.refiner.current_strategy
                molecules, gen_msg = self.generator.generate_batch(
                    seeds, batch_size, strategy
                )
                self.message_log.append(gen_msg)

                if not molecules:
                    logger.warning("No molecules generated, attempting retry...")
                    molecules, gen_msg = self.generator.generate_batch(
                        seeds, batch_size, "evolutionary"
                    )
                    self.message_log.append(gen_msg)

            # Step 2: Evaluator Agent
            logger.info("Step 2: Evaluator Agent - Evaluating molecules...")
            evaluated, eval_msg = self.evaluator.evaluate_batch(molecules, iteration)
            self.message_log.append(eval_msg)

            if not evaluated:
                logger.warning("No molecules evaluated")
                return False, "Evaluation failed"

            # Step 3: Refiner Agent
            if self.ablation_mode == "no_refiner":
                logger.info("Step 3 skipped (ablation): no refiner analysis")
                return True, "Continuing optimization"

            logger.info("Step 3: Refiner Agent - Analyzing performance...")
            best_molecules = self.evaluator.get_top_molecules(10)
            refinements, ref_msg = self.refiner.analyze_and_refine(
                [m.to_dict() for m in best_molecules],
                iteration,
                max_iterations,
            )
            self.message_log.append(ref_msg)

            # Check termination
            should_terminate = refinements.get("should_terminate", False)
            reason = refinements.get("termination_reason", "")

            if should_terminate:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"OPTIMIZATION TERMINATING: {reason}")
                logger.info(f"{'=' * 60}\n")
                return False, reason

            return True, "Continuing optimization"

        except Exception as e:
            logger.error(f"Iteration failed: {e}")
            return False, f"Iteration error: {str(e)}"

    def get_final_results(self) -> Dict[str, Any]:
        """Get final optimization results."""

        best = self.evaluator.get_best_molecule()

        return {
            "best_molecule": best.to_dict() if best else None,
            "top_5": [m.to_dict() for m in self.evaluator.get_top_molecules(5)],
            "pareto_front": [m.to_dict() for m in self.evaluator.get_pareto_front()],
            "convergence_metrics": self.evaluator.get_convergence_metrics(),
            "agent_statistics": {
                "generator": self.generator.get_statistics(),
                "evaluator": self.evaluator.get_statistics(),
                "refiner": self.refiner.get_statistics(),
            },
            "total_iterations": self.iteration_count,
            "total_molecules_generated": self.generator.molecules_generated,
            "total_molecules_evaluated": self.evaluator.molecules_evaluated,
        }