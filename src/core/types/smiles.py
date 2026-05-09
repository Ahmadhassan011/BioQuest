"""
SMILES validation and types.
"""

from typing import Optional
from rdkit import Chem


def validate_smiles(smiles: str) -> bool:
    """
    Validate SMILES string.

    Args:
        smiles: SMILES string to validate

    Returns:
        True if valid, False otherwise
    """
    if not smiles:
        return False

    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Canonicalize SMILES string.

    Args:
        smiles: Input SMILES

    Returns:
        Canonical SMILES or None if invalid
    """
    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def get_num_atoms(smiles: str) -> int:
    """
    Get number of atoms in molecule.

    Args:
        smiles: SMILES string

    Returns:
        Number of atoms, or 0 if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return mol.GetNumAtoms()
    except Exception:
        return 0