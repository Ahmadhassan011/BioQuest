"""
Convergence: Tracking and plateau detection for optimization.

This module contains pure business logic for detecting when optimization
has converged or should terminate.
"""

import logging
from typing import Dict, List
import numpy as np
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class OptimizationHistory:
    """Track optimization progress across iterations."""

    best_scores: List[float] = field(default_factory=list)
    mean_scores: List[float] = field(default_factory=list)
    max_scores: List[float] = field(default_factory=list)
    diversity_metrics: List[float] = field(default_factory=list)
    iterations: List[int] = field(default_factory=list)
    molecules_generated: List[int] = field(default_factory=list)

    def add_iteration(
        self,
        iteration: int,
        best_score: float,
        mean_score: float,
        max_score: float,
        diversity: float = 0.0,
        num_molecules: int = 0,
    ) -> None:
        """Add iteration to history."""
        self.iterations.append(iteration)
        self.best_scores.append(best_score)
        self.mean_scores.append(mean_score)
        self.max_scores.append(max_score)
        self.diversity_metrics.append(diversity)
        self.molecules_generated.append(num_molecules)


class ConvergenceTracker:
    """Track convergence and detect plateaus."""

    def __init__(
        self,
        window_size: int = 10,
        plateau_threshold: float = 0.001,
        patience: int = 20,
    ):
        """
        Initialize convergence tracker.

        Args:
            window_size: Window size for moving average
            plateau_threshold: Minimum improvement to avoid plateau
            patience: Number of iterations without improvement before stopping
        """
        self.window_size = window_size
        self.plateau_threshold = plateau_threshold
        self.patience = patience

        self.score_history: deque = deque(maxlen=window_size)
        self.best_score = -np.inf
        self.best_iteration = 0
        self.iterations_since_improvement = 0

    def update(self, score: float, iteration: int) -> None:
        """
        Update tracker with new score.

        Args:
            score: Current best score
            iteration: Current iteration number
        """
        self.score_history.append(score)

        if score > self.best_score:
            improvement = score - self.best_score
            self.best_score = score
            self.best_iteration = iteration
            self.iterations_since_improvement = 0

            if improvement > self.plateau_threshold:
                logger.info(
                    f"Iteration {iteration}: New best score {score:.4f} "
                    f"(improvement: {improvement:.4f})"
                )
            else:
                logger.debug(
                    f"Iteration {iteration}: Marginal improvement {improvement:.4f}"
                )
        else:
            self.iterations_since_improvement += 1

    def is_converged(self) -> bool:
        """
        Check if optimization has converged (plateau detected).

        Returns:
            True if plateau detected, False otherwise
        """
        if len(self.score_history) < self.window_size:
            return False

        # Calculate moving average
        moving_avg = np.mean(list(self.score_history))

        # Check if improvement is below threshold
        if self.best_score > 0:
            improvement_rate = (moving_avg - self.best_score) / (self.best_score + 1e-8)
            is_plateau = abs(improvement_rate) < self.plateau_threshold

            if is_plateau:
                logger.info(f"Convergence detected. Best score: {self.best_score:.4f}")
                return True

        return False

    def exceeded_patience(self) -> bool:
        """
        Check if iterations without improvement exceeded patience.

        Returns:
            True if patience exceeded, False otherwise
        """
        exceeded = self.iterations_since_improvement >= self.patience
        if exceeded:
            logger.info(
                f"Patience exceeded: {self.iterations_since_improvement} iterations "
                f"without improvement"
            )
        return exceeded

    def get_convergence_metrics(self) -> Dict[str, float]:
        """Get convergence metrics."""
        return {
            "best_score": self.best_score,
            "best_iteration": self.best_iteration,
            "iterations_since_improvement": self.iterations_since_improvement,
            "moving_average": float(np.mean(list(self.score_history)))
            if self.score_history
            else 0.0,
        }