"""
Evaluation Module: Multi-objective optimization, convergence tracking, and plateau detection.

This module provides:
- Multi-objective evaluation framework
- Pareto front generation and optimization
- Weighted sum optimization
- Convergence and plateau detection
- Performance metrics tracking
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
from collections import deque
import json

logger = logging.getLogger(__name__)


@dataclass
class MoleculeScore:
    """Container for molecule evaluation results."""

    smiles: str
    affinity: float = 0.0
    toxicity: float = 0.0
    qed: float = 0.0
    sa: float = 0.0
    composite_score: float = 0.0
    iteration: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "smiles": self.smiles,
            "affinity": self.affinity,
            "toxicity": self.toxicity,
            "qed": self.qed,
            "sa": self.sa,
            "composite_score": self.composite_score,
            "iteration": self.iteration,
        }


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


class MultiObjectiveEvaluator:
    """Evaluates molecules on multiple objectives."""

    def __init__(self, objective_weights: Dict[str, float]):
        """
        Initialize multi-objective evaluator.

        Args:
            objective_weights: Dictionary of {objective_name: weight}
        """
        self.objective_weights = objective_weights

        # Normalize weights
        total = sum(objective_weights.values())
        if total > 0:
            self.normalized_weights = {
                k: v / total for k, v in objective_weights.items()
            }
        else:
            self.normalized_weights = {
                k: 1.0 / len(objective_weights) for k in objective_weights.keys()
            }

        logger.info(
            f"MultiObjectiveEvaluator initialized with weights: {self.normalized_weights}"
        )

    def evaluate_weighted_sum(
        self,
        properties: Dict[str, float],
    ) -> float:
        """
        Calculate weighted sum score for molecule.

        Args:
            properties: Dictionary of {property_name: value}
                - Higher is better for: affinity, qed, sa
                - Lower is better for: toxicity (inverted in calculation)

        Returns:
            Composite score (0-1 range)
        """
        score = 0.0

        for objective, weight in self.normalized_weights.items():
            if objective not in properties:
                continue

            value = properties[objective]

            if objective == "toxicity":
                # Invert toxicity (lower is better)
                score += weight * (1.0 - value)
            else:
                # For affinity, qed, sa (higher is better)
                score += weight * value

        return np.clip(score, 0.0, 1.0)

    def calculate_pareto_front(
        self,
        molecules: List[Dict[str, float]],
    ) -> List[Dict[str, float]]:
        """
        Calculate Pareto front from molecule population.

        Args:
            molecules: List of dictionaries with properties

        Returns:
            List of molecules on Pareto front
        """
        if not molecules:
            return []

        objectives = ["affinity", "qed", "sa"]

        # Convert toxicity to negative (higher is better)
        objectives_to_check = []
        for mol in molecules:
            adjusted = mol.copy()
            adjusted["toxicity"] = 1.0 - adjusted.get("toxicity", 0.5)
            objectives_to_check.append(adjusted)

        # Find Pareto front
        pareto_front = []
        for i, mol_i in enumerate(objectives_to_check):
            dominated = False

            for j, mol_j in enumerate(objectives_to_check):
                if i == j:
                    continue

                # Check if mol_j dominates mol_i
                dominates = True
                for obj in objectives:
                    if mol_j.get(obj, 0.0) < mol_i.get(obj, 0.0):
                        dominates = False
                        break

                if dominates:
                    # Check if at least one objective is strictly better
                    strictly_better = any(
                        mol_j.get(obj, 0.0) > mol_i.get(obj, 0.0) for obj in objectives
                    )
                    if strictly_better:
                        dominated = True
                        break

            if not dominated:
                pareto_front.append(molecules[i])

        return pareto_front if pareto_front else [molecules[0]]


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


class OptimizationEvaluator:
    """Main evaluation orchestrator combining multi-objective and convergence tracking."""

    def __init__(
        self,
        objective_weights: Dict[str, float],
        plateau_threshold: float = 0.001,
        patience: int = 20,
    ):
        """
        Initialize optimization evaluator.

        Args:
            objective_weights: Dictionary of objective weights
            plateau_threshold: Threshold for plateau detection
            patience: Iterations without improvement before stopping
        """
        self.multi_objective = MultiObjectiveEvaluator(objective_weights)
        self.convergence = ConvergenceTracker(
            plateau_threshold=plateau_threshold,
            patience=patience,
        )
        self.history = OptimizationHistory()
        self.evaluated_molecules: Dict[str, MoleculeScore] = {}
        self.iteration_count = 0

    def evaluate_molecule(
        self,
        smiles: str,
        properties: Dict[str, float],
        iteration: int = 0,
    ) -> MoleculeScore:
        """
        Evaluate single molecule.

        Args:
            smiles: Molecule SMILES string
            properties: Dictionary of molecular properties
            iteration: Current iteration number

        Returns:
            MoleculeScore object
        """
        composite_score = self.multi_objective.evaluate_weighted_sum(properties)

        mol_score = MoleculeScore(
            smiles=smiles,
            affinity=properties.get("affinity", 0.5),
            toxicity=properties.get("toxicity", 0.5),
            qed=properties.get("qed", 0.5),
            sa=properties.get("sa", 0.5),
            composite_score=composite_score,
            iteration=iteration,
        )

        self.evaluated_molecules[smiles] = mol_score
        return mol_score

    def evaluate_population(
        self,
        population: List[Dict[str, float]],
        iteration: int = 0,
    ) -> List[MoleculeScore]:
        """
        Evaluate population of molecules.

        Args:
            population: List of {smiles: ..., affinity: ..., ...}
            iteration: Current iteration number

        Returns:
            List of MoleculeScore objects
        """
        scores = []
        for mol_data in population:
            smiles = mol_data.get("smiles")
            if smiles:
                properties = {k: v for k, v in mol_data.items() if k != "smiles"}
                score = self.evaluate_molecule(smiles, properties, iteration)
                scores.append(score)

        return scores

    def update_iteration(
        self,
        scores: List[MoleculeScore],
        iteration: int,
    ) -> None:
        """
        Update history after iteration.

        Args:
            scores: List of molecule scores from iteration
            iteration: Current iteration number
        """
        if not scores:
            return

        self.iteration_count = iteration

        # Calculate statistics
        composite_scores = [s.composite_score for s in scores]
        best_score = max(composite_scores)
        mean_score = np.mean(composite_scores)
        max_score = np.max(composite_scores)

        # Calculate diversity (unique molecules)
        unique_molecules = len(set(s.smiles for s in scores))

        # Update tracking
        self.convergence.update(best_score, iteration)
        self.history.add_iteration(
            iteration=iteration,
            best_score=best_score,
            mean_score=mean_score,
            max_score=max_score,
            diversity=unique_molecules / len(scores) if scores else 0.0,
            num_molecules=len(scores),
        )

        logger.info(
            f"Iteration {iteration}: Best={best_score:.4f}, "
            f"Mean={mean_score:.4f}, Unique={unique_molecules}/{len(scores)}"
        )

    def get_best_molecule(self) -> Optional[MoleculeScore]:
        """Get best evaluated molecule."""
        if not self.evaluated_molecules:
            return None
        return max(
            self.evaluated_molecules.values(),
            key=lambda x: x.composite_score,
        )

    def get_top_molecules(self, k: int = 10) -> List[MoleculeScore]:
        """Get top k molecules."""
        sorted_mols = sorted(
            self.evaluated_molecules.values(),
            key=lambda x: x.composite_score,
            reverse=True,
        )
        return sorted_mols[:k]

    def should_terminate(self, max_iterations: int) -> Tuple[bool, str]:
        """
        Determine if optimization should terminate.

        Args:
            max_iterations: Maximum allowed iterations

        Returns:
            Tuple of (should_terminate, reason)
        """
        if self.iteration_count >= max_iterations:
            return True, f"Maximum iterations ({max_iterations}) reached"

        if self.convergence.is_converged():
            return True, "Convergence plateau detected"

        if self.convergence.exceeded_patience():
            return True, f"Patience exceeded ({self.convergence.patience} iterations)"

        return False, "Continuing optimization"

    def get_convergence_metrics(self) -> Dict:
        """Get complete convergence metrics."""
        return {
            "convergence": self.convergence.get_convergence_metrics(),
            "history": {
                "iterations": self.history.iterations,
                "best_scores": self.history.best_scores,
                "mean_scores": self.history.mean_scores,
                "diversity": self.history.diversity_metrics,
            },
            "statistics": {
                "total_evaluated": len(self.evaluated_molecules),
                "best_score": self.convergence.best_score,
                "current_iteration": self.iteration_count,
            },
        }

    def get_pareto_front(self, top_k: int = 50) -> List[MoleculeScore]:
        """
        Get Pareto front from top molecules.

        Args:
            top_k: Number of top molecules to consider

        Returns:
            List of molecules on Pareto front
        """
        top_molecules = self.get_top_molecules(top_k)
        mol_dicts = [
            {
                "smiles": m.smiles,
                "affinity": m.affinity,
                "toxicity": m.toxicity,
                "qed": m.qed,
                "sa": m.sa,
            }
            for m in top_molecules
        ]

        pareto_indices = self.multi_objective.calculate_pareto_front(mol_dicts)

        # Map back to MoleculeScore objects
        pareto_smiles = {p["smiles"] for p in pareto_indices}
        return [m for m in top_molecules if m.smiles in pareto_smiles]

    def save_history(self, filepath: str) -> None:
        """Save optimization history to JSON file."""
        try:
            data = {
                "convergence_metrics": self.get_convergence_metrics(),
                "top_molecules": [m.to_dict() for m in self.get_top_molecules(100)],
                "pareto_front": [m.to_dict() for m in self.get_pareto_front()],
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Optimization history saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
