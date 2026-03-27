"""
Data Preparation Module: Custom datasets and dataset preparers.

DATA CACHING:
    Processed datasets are automatically cached to <project_root>/data/processed/
    to avoid re-featurization. Subsequent runs load from cache unless cache is cleared.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem import Descriptors

from tdc.single_pred import ADME

from src.data.loaders import TDCDataLoader
from src.data.storage import DataCache
from src.models.featurization import MolecularFeaturizer

logger = logging.getLogger(__name__)


class DTIGraphDataset(Dataset):
    """Custom Dataset for DTI graph-based data."""

    def __init__(self, data_list: List[Any]):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


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


class DTIDatasetPreparer:
    """
    Prepares Drug-Target Interaction data from PyTDC for training.

    Features:
    - Automatic featurization of molecules and proteins
    - Train/val/test splitting
    - Protein sequence truncation to manage memory
    """

    def __init__(self, featurizer: Optional[MolecularFeaturizer] = None):
        """
        Initialize dataset preparer.

        Args:
            featurizer: Molecular featurizer instance
        """
        self.featurizer = featurizer or MolecularFeaturizer()
        self.tdc_loader = TDCDataLoader()

    def prepare_dti_dataset(
        self,
        dataset_name: str = "DAVIS",
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_prot_len: int = 512,
    ) -> Tuple[List[Any], Dict, Dict]:
        """
        Prepare DTI dataset from PyTDC with local caching.

        Args:
            dataset_name: PyTDC dataset name
            val_split: Validation set fraction
            test_split: Test set fraction
            max_prot_len: Maximum length for protein sequences to truncate to.

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
                raise ValueError(f"Failed to load {dataset_name} dataset")

            smiles_list = dti_data["Drug"].tolist()
            protein_seqs = dti_data["Target"].tolist()
            affinities = dti_data["Y"].tolist()

            logger.info(f"Loaded {len(smiles_list)} DTI interactions")

            logger.info(
                f"Featurizing molecules and proteins (truncating proteins to {max_prot_len})..."
            )
            data_list = []
            protein_handler = self.tdc_loader.protein_handler
            for smiles, protein_seq, affinity in zip(
                smiles_list, protein_seqs, affinities
            ):
                mol_graph = self.featurizer.featurize_molecule_graph(smiles)
                if mol_graph is None:
                    continue

                truncated_protein_seq = protein_seq[:max_prot_len]

                try:
                    prot_indices = protein_handler.prepare_protein_indices(
                        truncated_protein_seq
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

            all_affinities = np.array([data.y.item() for data in data_list])
            min_aff, max_aff = np.min(all_affinities), np.max(all_affinities)
            for data in data_list:
                data.y = (data.y - min_aff) / (max_aff - min_aff + 1e-10)

            n_samples = len(data_list)
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
                "atom_feature_dim": data_list[0].x.shape[1],
                "protein_max_length": max_prot_len,
                "affinity_min": float(min_aff),
                "affinity_max": float(max_aff),
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

        except Exception as e:
            logger.error(f"Failed to prepare DTI dataset: {e}", exc_info=True)
            raise


class Tox21DatasetPreparer:
    """Prepares Tox21 data from PyTDC for training the toxicity model."""

    def __init__(self, featurizer: Optional[MolecularFeaturizer] = None):
        self.featurizer = featurizer or MolecularFeaturizer()
        self.tdc_loader = TDCDataLoader()

    def prepare_tox21_dataset(
        self,
        assay: str = "NR-AR",
        val_split: float = 0.1,
        test_split: float = 0.1,
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

            smiles_list = assay_data["Drug"].tolist()
            labels = assay_data["Y"].to_numpy(dtype=np.float32)

            logger.info(f"Featurizing {len(smiles_list)} molecules for {assay}...")
            mol_features = self.featurizer.batch_featurize_molecules(smiles_list)

            n_samples = len(mol_features)
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
        except Exception as e:
            logger.error(f"Failed to prepare Tox21 dataset: {e}", exc_info=True)
            raise


TDC_PROPERTY_DATA_MAPPING = {
    "lipophilicity_astrazeneca": ADME,
    "Caco2_Wang": ADME,
    "HIA_Hou": ADME,
    "BBB_Martins": ADME,
    "PPB_Watanabe": ADME,
    "VDss_Lombardo": ADME,
    "Solubility_AqSolDB": ADME,
}


class PropertyDatasetPreparer:
    """
    Prepares multi-task property prediction data.
    Calculates QED, SA (TPSA proxy), LogP, MW for molecules.
    """

    def __init__(self, featurizer: Optional[MolecularFeaturizer] = None):
        self.featurizer = featurizer or MolecularFeaturizer()

        self.logp_min, self.logp_max = -3.0, 5.0
        self.mw_min, self.mw_max = 50.0, 800.0
        self.tpsa_min, self.tpsa_max = 0.0, 200.0

    def _normalize_logp(self, logp: float) -> float:
        """Normalizes LogP to range [-1, 1]."""
        return 2 * ((logp - self.logp_min) / (self.logp_max - self.logp_min)) - 1

    def _normalize_mw(self, mw: float) -> float:
        """Normalizes Molecular Weight to range [0, 1]."""
        return (mw - self.mw_min) / (self.mw_max - self.mw_min)

    def _normalize_tpsa(self, tpsa: float) -> float:
        """Normalizes TPSA to range [0, 1]."""
        return (tpsa - self.tpsa_min) / (self.tpsa_max - self.tpsa_min)

    def prepare_property_dataset(
        self,
        dataset_name: str = "Lipophilicity_ID",
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
                    else:
                        targets_dict = targets
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
                    "Please add it to TDC_PROPERTY_DATA_MAPPING in src/data/data_preparation.py."
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
                all_sa.append(self._normalize_tpsa(tpsa))
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
                "sa_min_max": (self.tpsa_min, self.tpsa_max),
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

        except Exception as e:
            logger.error(f"Failed to prepare Property dataset: {e}", exc_info=True)
            raise


class VAEDatasetPreparer:
    """Prepares ChEMBL data from PyTDC for training the VAE."""

    def __init__(self):
        self.tdc_loader = TDCDataLoader()
        self.smiles_chars = "CNOPSFClBrI()[]#=%\\C@H"
        self.char_to_idx = {c: i for i, c in enumerate(self.smiles_chars)}

    def smiles_to_indices(self, smiles: str, max_len: int = 100) -> np.ndarray:
        """Convert SMILES string to an array of indices."""
        indices = [self.char_to_idx.get(c, 0) for c in smiles]
        padded = np.zeros(max_len, dtype=int)
        padded[: len(indices)] = indices[:max_len]
        return padded

    def prepare_vae_dataset(
        self,
        sample_frac: float = 0.05,
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_smiles_len: int = 100,
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """Prepare ChEMBL dataset for VAE training with local caching."""
        suffix = f"frac_{sample_frac}_len_{max_smiles_len}"
        if DataCache.has_processed_data(
            "ChEMBL_V29", preparer_type="vae", suffix=suffix
        ):
            logger.info(
                f"Loading ChEMBL VAE dataset from cache (sample_frac={sample_frac})..."
            )
            try:
                cached = DataCache.load_processed_data(
                    "ChEMBL_V29", preparer_type="vae", suffix=suffix
                )
                if cached is not None:
                    smiles_tensors, splits, metadata = cached
                    logger.info(
                        f"Dataset loaded from cache: train={len(splits['train'])}, "
                        f"val={len(splits['val'])}, test={len(splits['test'])}"
                    )
                    return smiles_tensors, splits, metadata
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}. Preparing fresh...")

        logger.info("Loading ChEMBL dataset for VAE pretraining...")
        try:
            chembl_data = self.tdc_loader.load_chembl_data(sample_frac=sample_frac)
            smiles_list = chembl_data["smiles"].tolist()

            logger.info(f"Tokenizing {len(smiles_list)} SMILES strings...")

            tokenized_smiles = []
            for s in smiles_list:
                if isinstance(s, str):
                    tokenized_smiles.append(
                        self.smiles_to_indices(s, max_len=max_smiles_len)
                    )
                else:
                    logger.warning(f"Skipping invalid SMILES entry (not a string): {s}")

            smiles_tensors = np.array(tokenized_smiles)

            if len(smiles_tensors) == 0:
                raise ValueError("No valid SMILES strings found in the dataset.")

            n_samples = len(smiles_tensors)
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            test_size, val_size = (
                int(n_samples * test_split),
                int(n_samples * val_split),
            )
            test_idx, val_idx, train_idx = (
                indices[:test_size],
                indices[test_size : test_size + val_size],
                indices[test_size + val_size :],
            )
            splits = {"train": train_idx, "val": val_idx, "test": test_idx}

            metadata = {
                "dataset_name": "ChEMBL",
                "total_samples": n_samples,
                "max_smiles_len": max_smiles_len,
            }
            logger.info(
                f"ChEMBL dataset ready: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
            )

            DataCache.save_processed_data(
                smiles_tensors,
                splits,
                metadata,
                "ChEMBL_V29",
                preparer_type="vae",
                suffix=suffix,
            )

            return smiles_tensors, splits, metadata
        except Exception as e:
            logger.error(f"Failed to prepare VAE dataset: {e}", exc_info=True)
            raise
