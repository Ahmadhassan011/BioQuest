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

    Uses sample-count-balanced assignment: scaffolds are greedily assigned
    to splits until the desired fraction of total samples is reached.
    This prevents small scaffold groups from dominating split sizes.

    Args:
        smiles_list: List of SMILES strings.
        val_frac: Fraction of samples for validation.
        test_frac: Fraction of samples for test.
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

    n_total = len(smiles_list)
    test_target = int(n_total * test_frac)
    val_target = int(n_total * val_frac)

    test_idx: List[int] = []
    val_idx: List[int] = []
    train_idx: List[int] = []

    for scaf in scaffolds:
        idx_list = scaffold_to_indices[scaf]
        if len(test_idx) < test_target:
            test_idx.extend(idx_list)
        elif len(val_idx) < val_target:
            val_idx.extend(idx_list)
        else:
            train_idx.extend(idx_list)

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
    data_type: str  # "dti", "tox21", "property", "vae"

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