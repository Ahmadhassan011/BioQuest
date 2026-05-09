"""Property prediction dataset preparation."""

import logging
from typing import Dict, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem import Descriptors

from tdc.single_pred import ADME

from infra.exceptions import DataError, DataProcessingError
from ..storage import DataCache
from ...models.featurization import MolecularFeaturizer
from .base import BasePreparer

logger = logging.getLogger(__name__)


TDC_PROPERTY_DATA_MAPPING = {
    "Lipophilicity_AstraZeneca": ADME,
    "lipophilicity_astrazeneca": ADME,
    "Caco2_Wang": ADME,
    "HIA_Hou": ADME,
    "BBB_Martins": ADME,
    "PPB_Watanabe": ADME,
    "VDss_Lombardo": ADME,
    "Solubility_AqSolDB": ADME,
}


class PropertyPredictionDataset(Dataset):
    """Custom Dataset for multi-task property prediction."""

    def __init__(self, features: torch.Tensor, targets: Dict[str, torch.Tensor]):
        self.features = features
        self.targets = targets
        self.num_samples = len(features)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample_targets = {task: self.targets[task][idx] for task in self.targets}
        return self.features[idx], sample_targets


class PropertyDatasetPreparer(BasePreparer):
    """
    Prepares multi-task property prediction data.
    Calculates QED, SA (TPSA proxy), LogP, MW for molecules.
    """

    def __init__(self, featurizer: Optional[MolecularFeaturizer] = None):
        super().__init__()
        self.featurizer = featurizer or MolecularFeaturizer()

        self.logp_min, self.logp_max = -3.0, 5.0
        self.mw_min, self.mw_max = 50.0, 800.0
        self.tpsa_min, self.tpsa_max = 0.0, 200.0

    def prepare(self, **kwargs) -> Tuple[Any, Dict, Dict]:
        features, targets_dict, splits, metadata = self.prepare_property_dataset(**kwargs)
        return (features, targets_dict), splits, metadata

    def _normalize_logp(self, logp: float) -> float:
        """Normalizes LogP to range [-1, 1] using log transform for better distribution."""
        logp_shifted = logp + 3.0
        return 2 * (np.log1p(logp_shifted) / np.log1p(8.0)) - 1

    def _normalize_mw(self, mw: float) -> float:
        """Normalizes Molecular Weight to range [0, 1]."""
        return (mw - self.mw_min) / (self.mw_max - self.mw_min)

    def _normalize_tpsa(self, tpsa: float) -> float:
        """Normalizes TPSA to range [0, 1]."""
        return (tpsa - self.tpsa_min) / (self.tpsa_max - self.tpsa_min)

    def prepare_property_dataset(
        self,
        dataset_name: str = "Lipophilicity_AstraZeneca",
        val_split: float = 0.1,
        test_split: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict, Dict]:
        """
        Prepare property prediction dataset with local caching.

        Args:
            dataset_name: PyTDC dataset name (e.g., 'Lipophilicity_ID')
            val_split: Validation set fraction
            test_split: Test set fraction

        Returns:
            Tuple of (features, targets_dict, splits, metadata)
            targets_dict contains { "qed": ..., "sa": ..., "logp": ..., "mw": ... }
        """
        cache_suffix = f"props_{dataset_name}"
        if DataCache.has_processed_data(
            "Properties", preparer_type="properties", suffix=cache_suffix
        ):
            logger.info(f"Loading Property dataset '{dataset_name}' from cache...")
            try:
                cached = DataCache.load_processed_data(
                    "Properties", preparer_type="properties", suffix=cache_suffix
                )
                if cached is not None:
                    features, targets, splits, metadata = cached
                    if isinstance(targets, list):
                        targets_dict = {
                            "qed": torch.tensor(targets[0], dtype=torch.float),
                            "sa": torch.tensor(targets[1], dtype=torch.float),
                            "logp": torch.tensor(targets[2], dtype=torch.float),
                            "mw": torch.tensor(targets[3], dtype=torch.float),
                        }
                    elif isinstance(targets, dict):
                        targets_dict = targets
                    else:
                        targets_dict = {
                            "qed": torch.tensor(targets[0], dtype=torch.float),
                            "sa": torch.tensor(targets[1], dtype=torch.float),
                            "logp": torch.tensor(targets[2], dtype=torch.float),
                            "mw": torch.tensor(targets[3], dtype=torch.float),
                        }
                    logger.info(
                        f"Dataset loaded from cache: train={len(splits['train'])}, "
                        f"val={len(splits['val'])}, test={len(splits['test'])}"
                    )
                    return features, targets_dict, splits, metadata
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}. Preparing fresh...")

        logger.info(f"Loading property dataset: {dataset_name} from PyTDC...")
        try:
            if dataset_name in TDC_PROPERTY_DATA_MAPPING:
                tdc_class = TDC_PROPERTY_DATA_MAPPING[dataset_name]
                data_loader = tdc_class(name=dataset_name)
                property_data = data_loader.get_data()
            else:
                raise ValueError(
                    f"Unknown property prediction dataset: {dataset_name}. "
                    "Please add it to TDC_PROPERTY_DATA_MAPPING in src/data/preparation/property.py."
                )

            if property_data is None or len(property_data) == 0:
                raise ValueError(f"Failed to load {dataset_name} dataset from PyTDC")

            smiles_list = property_data["Drug"].tolist()

            all_features = []
            all_qed, all_sa, all_logp, all_mw = [], [], [], []

            logger.info(
                f"Featurizing and calculating properties for {len(smiles_list)} molecules..."
            )
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                qed = Descriptors.qed(mol)
                tpsa = Descriptors.TPSA(mol)
                logp = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)

                mol_features = self.featurizer.batch_featurize_molecules([smiles])
                if mol_features is None:
                    continue

                all_features.append(torch.tensor(mol_features).squeeze(0))
                all_qed.append(qed)
                all_sa.append(1.0 - (np.log10(mol.GetNumAtoms() + 1) / 2.0))
                all_logp.append(self._normalize_logp(logp))
                all_mw.append(self._normalize_mw(mw))

            if not all_features:
                raise ValueError(
                    "No valid molecules found or featurized for property prediction."
                )

            features_tensor = torch.stack(all_features).float()

            targets_dict = {
                "qed": torch.tensor(all_qed, dtype=torch.float),
                "sa": torch.tensor(all_sa, dtype=torch.float),
                "logp": torch.tensor(all_logp, dtype=torch.float),
                "mw": torch.tensor(all_mw, dtype=torch.float),
            }

            n_samples = len(features_tensor)
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            test_size = int(n_samples * test_split)
            val_size = int(n_samples * val_split)

            test_idx = indices[:test_size]
            val_idx = indices[test_size : test_size + val_size]
            train_idx = indices[test_size + val_size :]

            splits = {"train": train_idx, "val": val_idx, "test": test_idx}

            metadata = {
                "dataset_name": dataset_name,
                "total_samples": n_samples,
                "mol_feature_dim": features_tensor.shape[1],
                "qed_min_max": (0.0, 1.0),
                "sa_min_max": (0.0, 1.0),
                "logp_min_max": (self.logp_min, self.logp_max),
                "mw_min_max": (self.mw_min, self.mw_max),
            }

            logger.info(
                f"Property dataset prepared: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
            )

            targets_for_cache = [all_qed, all_sa, all_logp, all_mw]
            DataCache.save_processed_data(
                (features_tensor, targets_for_cache),
                splits,
                metadata,
                "Properties",
                preparer_type="properties",
                suffix=cache_suffix,
            )

            return features_tensor, targets_dict, splits, metadata

        except DataError:
            raise
        except Exception as e:
            raise DataProcessingError(f"Failed to prepare Property dataset: {e}") from e
