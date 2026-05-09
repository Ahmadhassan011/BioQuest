"""Load and validate protein sequences and seed molecules."""

import logging
from typing import Any, Dict, List, Optional
import numpy as np

from ..constants import (
    DEFAULT_MAX_PROTEIN_LENGTH,
    DEFAULT_MAX_SMILES_LENGTH,
    sequence_to_indices,
    indices_to_array,
    validate_sequence,
)

logger = logging.getLogger(__name__)


class ProteinDataHandler:
    """Handles protein sequence loading, validation, and storage."""

    def __init__(self, max_sequence_length: int = DEFAULT_MAX_PROTEIN_LENGTH):
        """
        Initialize protein data handler.

        Args:
            max_sequence_length: Maximum allowed protein sequence length
        """
        self.max_sequence_length = max_sequence_length

    def validate_protein_sequence(self, sequence: str) -> str:
        """
        Validate and potentially truncate protein sequence format.

        Args:
            sequence: Protein sequence string

        Returns:
            The validated and potentially truncated sequence string.
            Raises ValueError if sequence contains invalid amino acids or is not a string.
        """
        return validate_sequence(sequence, self.max_sequence_length)

    def prepare_protein_indices(self, sequence: str, max_len: Optional[int] = None) -> np.ndarray:
        """Convert protein sequence to an array of indices.

        Delegates to constants.sequence_to_indices for single-source-of-truth logic.

        Args:
            sequence: Protein sequence string
            max_len: Maximum length for padding (defaults to max_sequence_length)

        Returns:
            A numpy array of amino acid indices.
        """
        if max_len is None:
            max_len = self.max_sequence_length
        validated = self.validate_protein_sequence(sequence)
        return indices_to_array(sequence_to_indices(validated, max_len))

    def prepare_protein_data(self, sequence: str) -> np.ndarray:
        """Alias for prepare_protein_indices for API compatibility."""
        return self.prepare_protein_indices(sequence)


class DrugDataHandler:
    """Handles drug/SMILES data processing."""

    def __init__(self, max_smiles_length: int = DEFAULT_MAX_SMILES_LENGTH):
        self.max_smiles_length = max_smiles_length

    def prepare_drug_data(self, smiles: str) -> Dict[str, Any]:
        return {"smiles": smiles, "length": len(smiles)}

    def prepare_molecular_features(self, smiles: str) -> Optional[np.ndarray]:
        return None


class PairDataHandler:
    """Handles drug-target pair data."""

    def prepare_pair_data(
        self, drug_data: Dict[str, Any], protein_data: np.ndarray, label: float
    ) -> Dict[str, Any]:
        return {"drug": drug_data, "protein": protein_data, "label": label}


class GraphDataHandler:
    """Handles graph data preparation."""

    def prepare_graph_data(
        self, drug_data: Dict[str, Any], protein_data: np.ndarray, pair_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"drug": drug_data, "protein": protein_data, "pair": pair_data}


class MoleculeSeedHandler:
    """Manages seed molecule SMILES strings."""

    def __init__(self):
        """Initialize molecule seed handler."""
        self.seeds: List[str] = []

    def add_seed(self, smiles: str) -> None:
        """
        Add a seed molecule SMILES string.

        Args:
            smiles: SMILES string representation
        """
        if isinstance(smiles, str) and len(smiles) > 0:
            self.seeds.append(smiles)
            logger.info(f"Added seed molecule: {smiles}")

    def add_seeds_from_list(self, smiles_list: List[str]) -> None:
        """
        Add multiple seed molecules.

        Args:
            smiles_list: List of SMILES strings
        """
        for smiles in smiles_list:
            self.add_seed(smiles)

    def get_seeds(self) -> List[str]:
        """Get all seed molecules."""
        return self.seeds.copy()

    def clear_seeds(self) -> None:
        """Clear all seeds."""
        self.seeds = []
