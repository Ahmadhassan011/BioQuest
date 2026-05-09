"""DTI (Drug-Target Interaction) dataset preparation."""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset

from infra.exceptions import DataError, DataProcessingError
from ..load.handlers import ProteinDataHandler
from ..load.tdc import TDCDataLoader
from ..storage import DataCache
from ...models.featurization import MolecularFeaturizer
from .base import BasePreparer, scaffold_split_indices

logger = logging.getLogger(__name__)


class DTIGraphDataset(Dataset):
    """Custom Dataset for DTI graph-based data."""

    def __init__(self, data_list: List[Any]):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class DTIDatasetPreparer(BasePreparer):
    """
    Prepares Drug-Target Interaction data from PyTDC for training.

    Features:
    - Automatic featurization of molecules and proteins
    - Train/val/test splitting
    - Protein sequence truncation to manage memory
    """

    def __init__(self, featurizer: Optional[MolecularFeaturizer] = None):
        super().__init__()
        self.featurizer = featurizer or MolecularFeaturizer()
        self.tdc_loader = TDCDataLoader()

    def prepare(self, **kwargs) -> Tuple[Any, Dict, Dict]:
        return self.prepare_dti_dataset(**kwargs)

    def prepare_dti_dataset(
        self,
        dataset_name: str = "DAVIS",
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_prot_len: int = 1024,
        use_scaffold_split: bool = False,
    ) -> Tuple[List[Any], Dict, Dict]:
        """
        Prepare DTI dataset from PyTDC with local caching.

        Args:
            dataset_name: PyTDC dataset name
            val_split: Validation set fraction
            test_split: Test set fraction
            max_prot_len: Maximum length for protein sequences to truncate to.
            use_scaffold_split: If True, split by Murcko scaffold (no scaffold overlap).

        Returns:
            Tuple of (data_list, splits, metadata)
        """
        cache_suffix = f"protlen_{max_prot_len}"

        if DataCache.has_processed_data(
            dataset_name, preparer_type="dti", suffix=cache_suffix
        ):
            logger.info(
                f"Loading {dataset_name} DTI dataset from cache (max_prot_len={max_prot_len})..."
            )
            try:
                data_list, splits, metadata = DataCache.load_processed_data(
                    dataset_name, preparer_type="dti", suffix=cache_suffix
                )
                logger.info(
                    f"Dataset loaded from cache: train={len(splits['train'])}, "
                    f"val={len(splits['val'])}, test={len(splits['test'])}"
                )
                return data_list, splits, metadata
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}. Preparing fresh...")

        logger.info(f"Loading {dataset_name} dataset from PyTDC...")

        try:
            dti_data = self.tdc_loader.load_dti_data(dataset_name)

            if dti_data is None or len(dti_data) == 0:
                raise DataError(f"Failed to load {dataset_name} dataset")

            original_len = len(dti_data)
            dti_data = dti_data[dti_data["Y"] < 10000].copy()
            filtered_len = len(dti_data)
            if filtered_len < original_len:
                logger.warning(
                    f"Filtered out {original_len - filtered_len} censored samples "
                    f"(Y=10000) from {dataset_name}"
                )

            smiles_list = dti_data["Drug"].tolist()
            protein_seqs = dti_data["Target"].tolist()
            affinities = dti_data["Y"].tolist()

            logger.info(f"Loaded {len(smiles_list)} DTI interactions after filtering")

            logger.info(
                f"Featurizing molecules and proteins (truncating proteins to {max_prot_len})..."
            )
            data_list = []
            successful_smiles = []
            protein_handler = ProteinDataHandler(max_sequence_length=max_prot_len)
            for smiles, protein_seq, affinity in zip(
                smiles_list, protein_seqs, affinities
            ):
                mol_graph = self.featurizer.featurize_molecule_graph(smiles)
                if mol_graph is None:
                    continue

                try:
                    prot_indices = protein_handler.prepare_protein_indices(
                        protein_seq
                    )

                    padded_prot_indices = np.zeros(max_prot_len, dtype=np.int64)
                    padded_prot_indices[: len(prot_indices)] = prot_indices

                except ValueError as e:
                    logger.warning(
                        f"Skipping DTI interaction due to invalid protein sequence: {e}"
                    )
                    continue

                data = mol_graph
                data.prot = torch.tensor(padded_prot_indices, dtype=torch.long)
                data.y = torch.tensor([affinity], dtype=torch.float)
                data_list.append(data)
                successful_smiles.append(smiles)

            n_samples = len(data_list)

            if use_scaffold_split:
                splits = scaffold_split_indices(
                    successful_smiles,
                    val_frac=val_split,
                    test_frac=test_split,
                )
            else:
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                test_size = int(n_samples * test_split)
                val_size = int(n_samples * val_split)
                test_idx = indices[:test_size]
                val_idx = indices[test_size : test_size + val_size]
                train_idx = indices[test_size + val_size :]
                splits = {"train": train_idx, "val": val_idx, "test": test_idx}

            train_affinities = np.array([data_list[i].y.item() for i in train_idx])

            train_log_affinities = np.log10(train_affinities + 1e-6)
            log_min = np.min(train_log_affinities)
            log_max = np.max(train_log_affinities)

            for data in data_list:
                y_log = np.log10(data.y.item() + 1e-6)
                data.y = torch.tensor(
                    [(y_log - log_min) / (log_max - log_min + 1e-10)],
                    dtype=torch.float,
                )

            metadata = {
                "dataset_name": dataset_name,
                "total_samples": n_samples,
                "atom_feature_dim": data_list[0].x.shape[1],
                "protein_max_length": max_prot_len,
                "affinity_transform": "log10",
                "affinity_log_min": float(log_min),
                "affinity_log_max": float(log_max),
                "affinity_original_min": float(np.min(train_affinities)),
                "affinity_original_max": float(np.max(train_affinities)),
            }

            logger.info(
                f"Dataset prepared: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
            )

            DataCache.save_processed_data(
                data_list,
                splits,
                metadata,
                dataset_name,
                preparer_type="dti",
                suffix=cache_suffix,
            )

            return data_list, splits, metadata

        except DataError:
            raise
        except Exception as e:
            raise DataProcessingError(f"Failed to prepare DTI dataset: {e}") from e
