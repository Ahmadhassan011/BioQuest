"""
Shared constants for protein sequence handling.

Single source of truth for protein-related constants across the codebase.
"""

from typing import List
import numpy as np

AMINO_ACIDS: List[str] = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]

AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}

VALID_AA_SET = set(AMINO_ACIDS) | {"X"}

DEFAULT_MAX_PROTEIN_LENGTH = 1024
DEFAULT_MAX_SMILES_LENGTH = 200
VAE_MAX_SMILES_LENGTH = 100

def create_aa_mapping() -> dict:
    """Get amino acid to index mapping."""
    return AA_TO_IDX.copy()

def validate_sequence(sequence: str, max_length: int = DEFAULT_MAX_PROTEIN_LENGTH) -> str:
    """
    Validate and truncate a protein sequence.

    Args:
        sequence: Protein sequence string
        max_length: Maximum allowed length

    Returns:
        Validated and truncated sequence

    Raises:
        ValueError: If sequence is not a string or contains invalid amino acids
    """
    if not isinstance(sequence, str):
        raise ValueError("Protein sequence must be a string")

    if len(sequence) > max_length:
        sequence = sequence[:max_length]

    sequence = sequence.upper()
    if not all(aa in VALID_AA_SET for aa in sequence):
        invalid = set(aa for aa in sequence if aa not in VALID_AA_SET)
        raise ValueError(f"Sequence contains invalid amino acids: {invalid}")

    return sequence

def sequence_to_indices(sequence: str, max_len: int) -> List[int]:
    """Convert protein sequence to index list with padding.

    Args:
        sequence: Protein sequence string
        max_len: Maximum length for padding/truncation

    Returns:
        List of amino acid indices (1-20, 0=padding)

    Raises:
        ValueError: If sequence contains invalid amino acids or is not a string
    """
    validated = validate_sequence(sequence, max_len)
    indices = [AA_TO_IDX.get(aa, 0) for aa in validated]
    if len(indices) < max_len:
        indices = indices + [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices


def indices_to_array(indices: List[int], dtype: type = np.int64) -> np.ndarray:
    """Convert a list of indices to a numpy array.

    Args:
        indices: List of amino acid indices
        dtype: Desired output dtype

    Returns:
        Numpy array of indices
    """
    return np.array(indices, dtype=dtype)