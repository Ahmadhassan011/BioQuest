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


def create_dataset(
    protein_sequence: str,
    seed_smiles: List[str],  # Currently not used for dataset creation in preparer
    objectives: Dict[
        str, float
    ],  # Not directly used for dataset creation here, but in optimization
    dataset_name: str = "DAVIS",  # Default DTI dataset
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_prot_len: int = 512,
) -> Tuple[Any, Dict, Dict]:  # Return type depends on the specific preparer
    """
    Factory function to create and prepare datasets based on input parameters.

    Args:
        protein_sequence: The protein sequence for DTI datasets.
        seed_smiles: Seed molecules for generation (not used for dataset prep here).
        objectives: Optimization objectives (not used for dataset prep here).
        dataset_name: Name of the dataset to prepare (e.g., "DAVIS" for DTI).
        val_split: Fraction of data for validation set.
        test_split: Fraction of data for test set.
        max_prot_len: Maximum protein sequence length for DTI featurization.

    Returns:
        A tuple containing (data_list, splits, metadata) from the appropriate preparer.

    Raises:
        ValueError: If an unsupported dataset type is requested.
    """
    logger.info(
        f"Creating dataset for protein: {protein_sequence[:20]}... with objectives: {objectives}"
    )

    # For now, we assume if protein_sequence is provided, we're doing DTI.
    # More sophisticated logic could be added here based on 'objectives' or a 'dataset_type' param.
    if protein_sequence:
        dti_preparer = DTIDatasetPreparer()
        # The DTIDatasetPreparer's prepare_dti_dataset already handles fetching
        # the dataset (e.g., DAVIS) and featurizing it based on its internal
        # logic and the provided max_prot_len.
        # The protein_sequence input to this factory is primarily for context
        # and could eventually be used to select/filter DTI data for a specific target.
        # For simplicity, we'll use a default DTI dataset name for now.
        return dti_preparer.prepare_dti_dataset(
            dataset_name=dataset_name,  # This will default to DAVIS
            val_split=val_split,
            test_split=test_split,
            max_prot_len=max_prot_len,
        )
    else:
        # Placeholder for other dataset types if needed in the future
        raise ValueError(
            "Currently, only DTI dataset creation based on protein sequence is supported via this factory."
        )


__all__ = ["data", "data_preparation", "cache", "DataCache", "create_dataset"]
