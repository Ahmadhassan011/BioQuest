"""Shared SMILES tokenizer for VAE training and inference.

Index 0 is reserved for PAD/UNK (padding and unknown characters).
Real tokens start at index 1.
"""

from typing import List
import numpy as np

PAD_TOKEN = 0

SMILES_TOKENS = [
    "Cl", "Br",
    "B", "C", "N", "O", "P", "S", "F", "I",
    "c", "n", "o", "p", "s",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "(", ")", "[", "]", "#", "=", "-", "+",
    "/", "\\", "@", "%", "H",
]
TOKEN_TO_IDX = {t: i + 1 for i, t in enumerate(SMILES_TOKENS)}
VOCAB_SIZE = len(SMILES_TOKENS) + 1


def smiles_to_indices(smiles: str, max_len: int = 100) -> np.ndarray:
    """Convert SMILES to indices via longest-match tokenization.

    Index 0 is PAD/UNK. Real tokens are 1-indexed.

    Args:
        smiles: SMILES string.
        max_len: Maximum sequence length (padded/truncated).

    Returns:
        Array of shape (max_len,) with token indices.
    """
    indices: List[int] = []
    i = 0
    while i < len(smiles):
        matched = False
        for token_len in (2, 1):
            if i + token_len <= len(smiles):
                token = smiles[i:i + token_len]
                if token in TOKEN_TO_IDX:
                    indices.append(TOKEN_TO_IDX[token])
                    i += token_len
                    matched = True
                    break
        if not matched:
            indices.append(0)
            i += 1
    padded = np.zeros(max_len, dtype=int)
    padded[:len(indices)] = indices[:max_len]
    return padded


def indices_to_smiles(indices: np.ndarray) -> str:
    """Convert token indices back to a SMILES string.

    Stops at the first PAD token (index 0).

    Args:
        indices: Array of token indices.

    Returns:
        Decoded SMILES string.
    """
    chars = []
    for idx in indices:
        if idx == 0:
            break
        if 1 <= idx <= len(SMILES_TOKENS):
            chars.append(SMILES_TOKENS[idx - 1])
    return "".join(chars)


def tokenize_smiles_list(smiles_list: List[str], max_len: int = 100) -> np.ndarray:
    """Batch tokenize a list of SMILES strings."""
    result = np.zeros((len(smiles_list), max_len), dtype=int)
    for i, smi in enumerate(smiles_list):
        result[i] = smiles_to_indices(smi, max_len)
    return result
