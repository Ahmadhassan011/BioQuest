"""VAE dataset preparation (ChEMBL)."""

import logging
from typing import Dict, Tuple, Any
import numpy as np

from infra.exceptions import DataError, DataProcessingError
from ..tokenizer import smiles_to_indices as tokenize, VOCAB_SIZE
from ..load.tdc import TDCDataLoader
from ..storage import DataCache
from .base import BasePreparer, scaffold_split_indices

logger = logging.getLogger(__name__)


class VAEDatasetPreparer(BasePreparer):
    """Prepares ChEMBL data from PyTDC for training the VAE."""

    def __init__(self):
        super().__init__()
        self.tdc_loader = TDCDataLoader()

    def prepare(self, **kwargs) -> Tuple[Any, Dict, Dict]:
        return self.prepare_vae_dataset(**kwargs)

    def prepare_vae_dataset(
        self,
        sample_frac: float = 0.05,
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_smiles_len: int = 100,
        use_scaffold_split: bool = False,
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """Prepare ChEMBL dataset for VAE training with local caching."""
        suffix = f"frac_{sample_frac}_len_{max_smiles_len}"
        if use_scaffold_split:
            suffix += "_scaffold"
        cache_name = "ChEMBL"
        if DataCache.has_processed_data(cache_name, preparer_type="vae", suffix=suffix):
            logger.info(
                f"Loading ChEMBL VAE dataset from cache (sample_frac={sample_frac})..."
            )
            try:
                cached = DataCache.load_processed_data(
                    cache_name, preparer_type="vae", suffix=suffix
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
            raw_smiles = chembl_data["smiles"].tolist()

            from rdkit import Chem
            smiles_list = []
            for s in raw_smiles:
                if isinstance(s, str):
                    mol = Chem.MolFromSmiles(s)
                    if mol is not None:
                        smiles_list.append(Chem.MolToSmiles(mol))
                else:
                    logger.warning(f"Skipping invalid SMILES entry (not a string): {s}")

            if not smiles_list:
                raise ValueError("No valid SMILES strings found.")

            logger.info(f"Tokenizing {len(smiles_list)} SMILES strings...")

            tokenized_smiles = np.array([
                tokenize(s, max_len=max_smiles_len) for s in smiles_list
            ])

            n_samples = len(tokenized_smiles)

            if use_scaffold_split:
                splits = scaffold_split_indices(
                    smiles_list, val_frac=val_split, test_frac=test_split,
                )
            else:
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
                "vocab_size": VOCAB_SIZE,
            }
            logger.info(
                f"ChEMBL dataset ready: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
            )

            DataCache.save_processed_data(
                tokenized_smiles,
                splits,
                metadata,
                cache_name,
                preparer_type="vae",
                suffix=suffix,
            )

            return tokenized_smiles, splits, metadata
        except DataError:
            raise
        except Exception as e:
            raise DataProcessingError(f"Failed to prepare VAE dataset: {e}") from e
