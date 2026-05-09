"""
Multi-objective molecule optimization orchestrator.

Combines objectives, convergence tracking, and evaluation logic.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .objectives import MultiObjectiveEvaluator
from .convergence import ConvergenceTracker, OptimizationHistory
from .pareto import get_pareto_front

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

        # Create adjusted list with toxicity inverted (higher is better)
        adjusted_dicts = []
        for mol in mol_dicts:
            adjusted = mol.copy()
            adjusted["toxicity"] = 1.0 - adjusted.get("toxicity", 0.5)
            adjusted_dicts.append(adjusted)

        pareto_indices = get_pareto_front(adjusted_dicts, ["affinity", "toxicity", "qed", "sa"])

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