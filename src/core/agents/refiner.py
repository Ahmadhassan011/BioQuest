"""
Refiner Agent: Analyzes convergence and adjusts generation strategy.

This agent decides WHEN to terminate and WHAT strategy to use.
It delegates HOW to the OptimizerEvaluator.
"""

import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime

from .messages import AgentMessage

logger = logging.getLogger(__name__)


class RefinerAgent:
    """
    Agent responsible for refining optimization strategy.

    Analyzes convergence metrics, adjusts generation parameters,
    and decides on termination conditions.
    """

    def __init__(self, evaluator):
        """
        Initialize refiner agent.

        Args:
            evaluator: OptimizationEvaluator instance (from core/optimization/)
        """
        self.evaluator = evaluator
        self.name = "RefinerAgent"
        self.refinement_history: List[Dict] = []
        self.current_strategy = "hybrid"

    def analyze_and_refine(
        self,
        best_molecules: List[Dict],
        iteration: int,
        max_iterations: int = 100,
    ) -> Tuple[Dict[str, Any], AgentMessage]:
        """
        Analyze performance and refine strategy.

        Args:
            best_molecules: Top performing molecules
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations

        Returns:
            Tuple of (strategy_updates, message_to_generator)
        """
        try:
            # Get convergence metrics
            convergence = self.evaluator.convergence.get_convergence_metrics()

            # Determine termination
            should_terminate, reason = self.evaluator.should_terminate(max_iterations)

            # Analyze convergence and adjust strategy
            if convergence["iterations_since_improvement"] > 5:
                # Switch strategy if stuck
                if self.current_strategy == "hybrid":
                    self.current_strategy = "evolutionary"
                elif self.current_strategy == "evolutionary":
                    self.current_strategy = "vae"
                else:
                    self.current_strategy = "hybrid"

                logger.info(f"Switching to {self.current_strategy} strategy")

            # Prepare refinement instructions
            refinements = {
                "strategy": self.current_strategy,
                "num_molecules": 50,
                "mutation_rate": 0.3
                + (convergence["iterations_since_improvement"] * 0.02),
                "should_terminate": should_terminate,
                "termination_reason": reason,
            }

            # Create message for generator
            message = AgentMessage(
                sender=self.name,
                receiver="GeneratorAgent",
                message_type="strategy_update",
                content=refinements,
            )

            self.refinement_history.append(
                {
                    "iteration": iteration,
                    "strategy": self.current_strategy,
                    "should_terminate": should_terminate,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            logger.info(
                f"Refiner analysis: iterations_since_improvement="
                f"{convergence['iterations_since_improvement']}, "
                f"termination={should_terminate} ({reason})"
            )

            return refinements, message

        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            message = AgentMessage(
                sender=self.name,
                receiver="GeneratorAgent",
                message_type="refinement_error",
                content={"error": str(e)},
            )
            return {"should_terminate": True, "termination_reason": str(e)}, message

    def get_statistics(self) -> Dict:
        """Get refiner statistics."""
        return {
            "agent": self.name,
            "refinement_rounds": len(self.refinement_history),
            "current_strategy": self.current_strategy,
            "history": self.refinement_history,
        }