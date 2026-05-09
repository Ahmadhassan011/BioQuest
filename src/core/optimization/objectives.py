"""
Objectives: Multi-objective evaluation with weighted sum scoring.

This module contains pure business logic for scoring molecules against
multiple objectives (affinity, toxicity, QED, SA).
"""

import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)


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

                # Check if mol_j dominates mol_i (Pareto dominance)
                # mol_j dominates mol_i if: mol_j >= mol_i on all objectives AND mol_j > mol_i on at least one
                dominates = True
                for obj in objectives:
                    # mol_j must be >= mol_i on all objectives (strictly greater or equal)
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