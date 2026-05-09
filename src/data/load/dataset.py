"""BioQuest dataset manager combining proteins, seeds, objectives, and DTI data."""

import logging
from typing import Dict, List, Optional
import numpy as np

from .handlers import ProteinDataHandler, MoleculeSeedHandler

logger = logging.getLogger(__name__)


class ObjectiveHandler:
    """Manages multi-objective optimization targets."""

    def __init__(self):
        """Initialize objective handler."""
        self.objectives: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}

    def add_objective(
        self, name: str, weight: float = 1.0, target: Optional[float] = None
    ) -> None:
        """
        Add optimization objective.

        Args:
            name: Objective name (e.g., "affinity", "toxicity", "qed", "sa")
            weight: Weight for multi-objective optimization (0-1 range recommended)
            target: Target value for objective (optional)
        """
        if weight < 0:
            raise ValueError("Objective weight must be non-negative")

        self.objectives[name] = target
        self.weights[name] = weight
        logger.info(f"Added objective '{name}' with weight {weight}")

    def get_objectives(self) -> Dict[str, float]:
        """Get all objectives and targets."""
        return self.objectives.copy()

    def get_weights(self) -> Dict[str, float]:
        """Get objective weights."""
        return self.weights.copy()

    def normalize_weights(self) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0.

        Returns:
            Dictionary of normalized weights
        """
        if not self.weights:
            return {}
        total = sum(self.weights.values())
        if total == 0:
            return {k: 1.0 / len(self.weights) for k in self.weights}
        return {k: v / total for k, v in self.weights.items()}


class BioQuestDataset:
    """Main dataset manager combining proteins, seeds, objectives, and DTI data."""

    def __init__(self):
        """Initialize BioQuest dataset."""
        self.protein_handler = ProteinDataHandler()
        self.seed_handler = MoleculeSeedHandler()
        self.objective_handler = ObjectiveHandler()
        self.protein_sequence: Optional[str] = None

    def set_protein_sequence(self, sequence: str) -> bool:
        """
        Set the target protein sequence.

        Args:
            sequence: Protein sequence string

        Returns:
            True if valid and set, False otherwise
        """
        try:
            validated_sequence = self.protein_handler.validate_protein_sequence(sequence)
            self.protein_sequence = validated_sequence.upper()
            logger.info(f"Set protein sequence (length: {len(self.protein_sequence)})")
            return True
        except ValueError as e:
            logger.warning(f"Invalid protein sequence: {e}")
            return False

    def add_seeds(self, smiles_list: List[str]) -> None:
        """Add seed molecules."""
        self.seed_handler.add_seeds_from_list(smiles_list)

    def add_objective(
        self, name: str, weight: float = 1.0, target: Optional[float] = None
    ) -> None:
        """
        Add optimization objective.

        Args:
            name: Objective name (e.g., "affinity", "toxicity", "qed", "sa")
            weight: Weight for multi-objective optimization (0-1 range recommended)
            target: Target value for objective (optional)
        """
        self.objective_handler.add_objective(name, weight=weight, target=target)

    def get_protein_embedding(self) -> Optional[np.ndarray]:
        """Get protein sequence embedding."""
        if self.protein_sequence:
            return self.protein_handler.prepare_protein_indices(self.protein_sequence)
        return None

    def get_seeds(self) -> List[str]:
        """Get seed molecules."""
        return self.seed_handler.get_seeds()

    def get_objectives(self) -> Dict[str, float]:
        """Get objectives."""
        return self.objective_handler.get_objectives()

    def get_objective_weights(self) -> Dict[str, float]:
        """Get normalized objective weights."""
        return self.objective_handler.normalize_weights()

    def summary(self) -> Dict:
        """
        Get dataset summary.

        Returns:
            Dictionary with configuration summary
        """
        return {
            "protein_length": len(self.protein_sequence)
            if self.protein_sequence
            else 0,
            "num_seeds": len(self.seed_handler.get_seeds()),
            "objectives": self.objective_handler.get_objectives(),
            "weights": self.objective_handler.normalize_weights(),
        }


def create_bioquest_dataset(
    protein_sequence: str,
    seed_smiles: List[str],
    objectives: Dict[str, float],
) -> BioQuestDataset:
    """
    Factory function to create and configure BioQuest dataset.

    Args:
        protein_sequence: Target protein sequence
        seed_smiles: List of seed molecule SMILES strings
        objectives: Dictionary of {objective_name: weight}

    Returns:
        Configured BioQuestDataset instance
    """
    dataset = BioQuestDataset()

    if not dataset.set_protein_sequence(protein_sequence):
        raise ValueError("Invalid protein sequence")

    dataset.add_seeds(seed_smiles)

    for obj_name, weight in objectives.items():
        dataset.add_objective(obj_name, weight)

    logger.info("Dataset created successfully")
    return dataset
