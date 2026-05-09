"""
Data preparation base classes and utilities.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Any, Dict, List, Union
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


def scaffold_split_indices(
    smiles_list: List[str],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Split molecules by Murcko scaffold so no scaffold appears in multiple splits.

    Args:
        smiles_list: List of SMILES strings.
        val_frac: Fraction of scaffolds for validation.
        test_frac: Fraction of scaffolds for test.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'train', 'val', 'test' index arrays.
    """
    rng = np.random.RandomState(seed)

    scaffold_to_indices: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            scaffold = Chem.MolToSmiles(mol)
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    scaffolds = list(scaffold_to_indices.keys())
    rng.shuffle(scaffolds)

    n = len(scaffolds)
    test_count = max(1, int(n * test_frac))
    val_count = max(1, int(n * val_frac))

    val_scaffolds = set(scaffolds[:val_count])
    train_scaffolds = set(scaffolds[test_count + val_count:])

    train_idx = []
    val_idx = []
    test_idx = []
    for scaf, idx_list in scaffold_to_indices.items():
        if scaf in train_scaffolds:
            train_idx.extend(idx_list)
        elif scaf in val_scaffolds:
            val_idx.extend(idx_list)
        else:
            test_idx.extend(idx_list)

    return {
        "train": np.array(train_idx, dtype=int),
        "val": np.array(val_idx, dtype=int),
        "test": np.array(test_idx, dtype=int),
    }


@dataclass
class DatasetResult:
    """Standardized container for dataset preparation results."""

    data: Union[List, np.ndarray, torch.Tensor]
    splits: Dict[str, Any]
    metadata: Dict[str, Any]
    data_type: str  # "dti", "tox21", "properties", "vae"

    @property
    def train_size(self) -> int:
        return len(self.splits.get("train", []))

    @property
    def val_size(self) -> int:
        return len(self.splits.get("val", []))

    @property
    def test_size(self) -> int:
        return len(self.splits.get("test", []))

    @property
    def total_samples(self) -> int:
        return self.metadata.get("total_samples", len(self.data))


class BasePreparer(ABC):
    """Base class for dataset preparers."""

    def __init__(self):
        """Initialize preparer."""
        self.name = self.__class__.__name__

    @abstractmethod
    def prepare(self, **kwargs) -> Tuple[Any, Dict, Dict]:
        """
        Prepare dataset.

        Returns:
            Tuple of (data, splits, metadata)
        """
        pass

    def validate_data(self, data: Any) -> bool:
        """Validate prepared data."""
        if data is None:
            return False
        if hasattr(data, '__len__') and len(data) == 0:
            return False
        return True