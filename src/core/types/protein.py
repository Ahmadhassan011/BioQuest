"""
Protein sequence validation and types.
"""

from typing import List


AMINO_ACIDS = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
]

AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}


def validate_protein_sequence(
    sequence: str,
    max_length: int = 1024,
) -> str:
    """
    Validate protein sequence.

    Args:
        sequence: Protein sequence string
        max_length: Maximum allowed length

    Returns:
        Uppercased, validated sequence

    Raises:
        ValueError: If sequence contains invalid amino acids
    """
    if not sequence:
        return ""

    sequence = sequence.upper()

    invalid_chars = set(sequence) - set(AMINO_ACIDS)
    if invalid_chars:
        raise ValueError(
            f"Invalid amino acids in sequence: {invalid_chars}"
        )

    if len(sequence) > max_length:
        sequence = sequence[:max_length]

    return sequence


def sequence_to_indices(sequence: str, max_len: int) -> List[int]:
    """
    Convert protein sequence to index list.

    Args:
        sequence: Protein sequence string
        max_len: Maximum length for padding

    Returns:
        List of amino acid indices (1-20, 0=padding)
    """
    validated = validate_protein_sequence(sequence, max_len)
    indices = [AA_TO_IDX.get(aa, 0) for aa in validated]

    if len(indices) < max_len:
        indices = indices + [0] * (max_len - len(indices))

    return indices