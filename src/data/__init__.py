"""Data module."""

import logging
from typing import Any, Dict, List, Tuple
from . import loaders, preparers, storage
from .storage import DataCache
from .preparers import (
    DTIDatasetPreparer,
    PropertyDatasetPreparer,
    Tox21DatasetPreparer,
    VAEDatasetPreparer,
)

logger = logging.getLogger(__name__)


def create_dti_dataset(
    protein_sequence: str,
    seed_smiles: List[str],
    objectives: Dict[str, float],
    dataset_name: str = "DAVIS",
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_prot_len: int = 512,
) -> Tuple[Any, Dict, Dict]:
    """
    Factory function to create and prepare DTI datasets.

    Args:
        protein_sequence: The protein sequence for DTI datasets.
        seed_smiles: Seed molecules for generation (not used for dataset prep).
        objectives: Optimization objectives (not used for dataset prep).
        dataset_name: Name of the dataset to prepare (default: "DAVIS").
        val_split: Fraction of data for validation set.
        test_split: Fraction of data for test set.
        max_prot_len: Maximum protein sequence length.

    Returns:
        A tuple containing (data_list, splits, metadata) from DTIDatasetPreparer.
    """
    logger.info(
        f"Creating DTI dataset for protein: {protein_sequence[:20]}... with objectives: {objectives}"
    )

    if protein_sequence:
        dti_preparer = DTIDatasetPreparer()
        return dti_preparer.prepare_dti_dataset(
            dataset_name=dataset_name,
            val_split=val_split,
            test_split=test_split,
            max_prot_len=max_prot_len,
        )
    else:
        raise ValueError(
            "protein_sequence is required for DTI dataset creation."
        )


def create_bioquest_dataset(
    protein_sequence: str,
    seed_smiles: List[str],
    objectives: Dict[str, float],
):
    """
    Create and configure a BioQuestDataset instance for agent orchestration.

    This is different from create_dti_dataset which returns raw data tuples.
    This returns a BioQuestDataset object with protein, seeds, and objectives.

    Args:
        protein_sequence: Target protein sequence
        seed_smiles: List of seed molecule SMILES strings
        objectives: Dictionary of {objective_name: weight}

    Returns:
        Configured BioQuestDataset instance
    """
    from .loaders import create_bioquest_dataset as _create

    return _create(protein_sequence, seed_smiles, objectives)


__all__ = [
    "data",
    "data_preparation",
    "cache",
    "DataCache",
    "create_dti_dataset",
    "create_bioquest_dataset",
]
