"""Tox21 dataset preparation."""

import logging
from typing import Dict, Tuple, Optional, Any
import numpy as np

from rdkit import Chem

from infra.exceptions import DataError, DataProcessingError
from ..load.tdc import TDCDataLoader
from ..storage import DataCache
from ...models.featurization import MolecularFeaturizer
from .base import BasePreparer, scaffold_split_indices

logger = logging.getLogger(__name__)


class Tox21DatasetPreparer(BasePreparer):
    """Prepares Tox21 data from PyTDC for training the toxicity model."""

    def __init__(self, featurizer: Optional[MolecularFeaturizer] = None):
        super().__init__()
        self.featurizer = featurizer or MolecularFeaturizer()
        self.tdc_loader = TDCDataLoader()

    def prepare(self, **kwargs) -> Tuple[Any, Dict, Dict]:
        mol_features, labels, splits, metadata = self.prepare_tox21_dataset(**kwargs)
        return (mol_features, labels), splits, metadata

    def prepare_tox21_dataset(
        self,
        assay: str = "NR-AR",
        val_split: float = 0.1,
        test_split: float = 0.1,
        use_scaffold_split: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """Prepare a specific Tox21 assay dataset with local caching."""
        if DataCache.has_processed_data(
            "Tox21", preparer_type="toxicity", suffix=assay
        ):
            logger.info(f"Loading Tox21 {assay} dataset from cache...")
            try:
                cached = DataCache.load_processed_data(
                    "Tox21", preparer_type="toxicity", suffix=assay
                )
                if cached is not None:
                    data, splits, metadata = cached
                    mol_features, labels = data
                    logger.info(
                        f"Dataset loaded from cache: train={len(splits['train'])}, "
                        f"val={len(splits['val'])}, test={len(splits['test'])}"
                    )
                    return mol_features, labels, splits, metadata
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}. Preparing fresh...")

        logger.info(f"Loading Tox21 dataset for assay: {assay}...")
        try:
            tox_data = self.tdc_loader.load_tox21_data()
            filtered_data = tox_data[tox_data["assay"] == assay].copy()
            assay_data = filtered_data[["Drug", "Y"]].dropna()

            # Canonicalize and validate SMILES
            raw_smiles = assay_data["Drug"].tolist()
            labels = assay_data["Y"].to_numpy(dtype=np.float32)
            smiles_list = []
            kept_labels = []
            for smi, label in zip(raw_smiles, labels):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    smiles_list.append(Chem.MolToSmiles(mol))
                    kept_labels.append(label)
            labels = np.array(kept_labels, dtype=np.float32)

            original_count = len(assay_data)
            if len(smiles_list) < original_count:
                logger.warning(f"Filtered out {original_count - len(smiles_list)} invalid/non-canonical SMILES")

            logger.info(f"Featurizing {len(smiles_list)} molecules for {assay}...")
            mol_features = self.featurizer.batch_featurize_molecules(smiles_list)

            n_samples = len(mol_features)

            if use_scaffold_split:
                splits = scaffold_split_indices(
                    smiles_list,
                    val_frac=val_split,
                    test_frac=test_split,
                )
            else:
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                test_size = int(n_samples * test_split)
                val_size = int(n_samples * val_split)
                test_idx, val_idx, train_idx = (
                    indices[:test_size],
                    indices[test_size : test_size + val_size],
                    indices[test_size + val_size :],
                )
                splits = {"train": train_idx, "val": val_idx, "test": test_idx}

            metadata = {
                "dataset_name": "Tox21",
                "assay": assay,
                "total_samples": n_samples,
                "mol_feature_dim": mol_features.shape[1],
            }
            logger.info(
                f"Tox21 dataset ready: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
            )

            DataCache.save_processed_data(
                (mol_features, labels),
                splits,
                metadata,
                "Tox21",
                preparer_type="toxicity",
                suffix=assay,
            )

            return mol_features, labels, splits, metadata
        except DataError:
            raise
        except Exception as e:
            raise DataProcessingError(f"Failed to prepare Tox21 dataset: {e}") from e
