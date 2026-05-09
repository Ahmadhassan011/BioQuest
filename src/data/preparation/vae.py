"""VAE dataset preparation (ChEMBL)."""

import logging
from typing import Dict, Tuple, Any
import numpy as np

from infra.exceptions import DataError, DataProcessingError
from ..constants import VAE_MAX_SMILES_LENGTH
from ..load.tdc import TDCDataLoader
from ..storage import DataCache
from .base import BasePreparer

logger = logging.getLogger(__name__)


class VAEDatasetPreparer(BasePreparer):
    """Prepares ChEMBL data from PyTDC for training the VAE."""

    def __init__(self):
        super().__init__()
        self.tdc_loader = TDCDataLoader()
        self.smiles_tokens = [
            "Cl", "Br",
            "C", "N", "O", "P", "S", "F", "I",
            "(", ")", "[", "]", "#", "=", "%",
            "\\", "@", "H",
        ]
        self.token_to_idx = {t: i for i, t in enumerate(self.smiles_tokens)}

    @property
    def smiles_chars(self) -> str:
        return "".join(self.smiles_tokens)

    def prepare(self, **kwargs) -> Tuple[Any, Dict, Dict]:
        return self.prepare_vae_dataset(**kwargs)

    def smiles_to_indices(self, smiles: str, max_len: int = VAE_MAX_SMILES_LENGTH) -> np.ndarray:
        """Convert SMILES string to an array of indices using longest-match tokenization."""
        indices = []
        i = 0
        while i < len(smiles):
            matched = False
            for token_len in (2, 1):
                if i + token_len <= len(smiles):
                    token = smiles[i:i + token_len]
                    if token in self.token_to_idx:
                        indices.append(self.token_to_idx[token])
                        i += token_len
                        matched = True
                        break
            if not matched:
                indices.append(0)
                i += 1
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
                cache_name,
                preparer_type="vae",
                suffix=suffix,
            )

            return smiles_tensors, splits, metadata
        except DataError:
            raise
        except Exception as e:
            raise DataProcessingError(f"Failed to prepare VAE dataset: {e}") from e
