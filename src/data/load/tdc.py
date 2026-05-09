"""Load datasets from the Therapeutics Data Commons (TDC)."""

import logging
from typing import Optional

import pandas as pd

from infra.exceptions import DataError
from ..storage import DataCache
from .handlers import ProteinDataHandler

logger = logging.getLogger(__name__)


class TDCDataLoader:
    """Loads datasets from the Therapeutics Data Commons (TDC)."""

    def __init__(self):
        """Initialize TDC data loader."""
        self.data = None
        self.protein_handler = ProteinDataHandler()

    def load_dti_data(self, dataset_name: str = "DAVIS") -> pd.DataFrame:
        """
        Load DTI dataset from PyTDC (with local caching).

        Args:
            dataset_name: Name of DTI dataset (e.g., "DAVIS", "KIBA", "BindingDB")

        Returns:
            DataFrame with columns: drug_id, target_id, y (binding affinity)
        """
        cached_data = DataCache.load_raw_data(dataset_name)
        if cached_data is not None:
            logger.info(f"Loaded {dataset_name} from local cache")
            if not hasattr(cached_data, 'shape') and not hasattr(cached_data, 'columns'):
                logger.warning(f"Cached data for {dataset_name} has invalid format, re-downloading...")
            else:
                self.data = cached_data
                return self.data

        try:
            from tdc.multi_pred import DTI

            logger.info(f"Downloading {dataset_name} from PyTDC...")
            dti = DTI(name=dataset_name)
            self.data = dti.get_data()

            DataCache.save_raw_data(self.data, dataset_name)

            logger.info(f"Loaded {len(self.data)} DTI interactions from {dataset_name}")
            return self.data
        except ImportError:
            raise DataError("TDC package not installed. Run: pip install PyTDC")
        except Exception as e:
            raise DataError(f"Failed to load DTI data: {str(e)}") from e

    def load_tox21_data(self, assay: Optional[str] = None) -> pd.DataFrame:
        """
        Load Tox21 dataset from PyTDC (with local caching).

        Args:
            assay: Specific Tox21 assay to load (e.g., 'NR-AR'). If None, loads all.

        Returns:
            DataFrame with toxicity data.
        """
        if assay is None:
            cached_data = DataCache.load_raw_data("Tox21")
            if cached_data is not None:
                logger.info("Loaded Tox21 from local cache")
                if not hasattr(cached_data, 'shape') and not hasattr(cached_data, 'columns'):
                    logger.warning("Cached Tox21 data has invalid format, re-downloading...")
                else:
                    self.data = cached_data
                    return self.data
        else:
            cached_data = DataCache.load_raw_data(f"Tox21_{assay}")
            if cached_data is not None:
                logger.info(f"Loaded Tox21_{assay} from local cache")
                if not hasattr(cached_data, 'shape') and not hasattr(cached_data, 'columns'):
                    logger.warning(f"Cached Tox21_{assay} data has invalid format, re-downloading...")
                else:
                    self.data = cached_data
                    return self.data

        try:
            from tdc.single_pred import Tox

            if assay:
                logger.info(f"Downloading Tox21 from PyTDC for assay {assay}...")
                tox = Tox(name="Tox21", label_name=assay)
                self.data = tox.get_data()
                DataCache.save_raw_data(self.data, f"Tox21_{assay}")
                logger.info(
                    f"Loaded {len(self.data)} compounds for Tox21 assay {assay}."
                )
                return self.data
            else:
                logger.info("Downloading all Tox21 assays from PyTDC...")
                from tdc.utils import retrieve_label_name_list

                labels = retrieve_label_name_list("Tox21")
                all_data = []
                for label in labels:
                    tox = Tox(name="Tox21", label_name=label)
                    df = tox.get_data()
                    df["assay"] = label
                    all_data.append(df)

                self.data = pd.concat(all_data, ignore_index=True)
                DataCache.save_raw_data(self.data, "Tox21")
                logger.info(
                    f"Loaded {len(self.data)} data points for all {len(labels)} Tox21 assays."
                )
                return self.data
        except ImportError:
            raise DataError("TDC package not installed. Run: pip install PyTDC")
        except Exception as e:
            raise DataError(f"Failed to load Tox21 data: {str(e)}") from e

    def load_chembl_data(self, sample_frac: float = 0.1) -> pd.DataFrame:
        """
        Load ChEMBL dataset for molecule generation pretraining (with local caching).

        Args:
            sample_frac: Fraction of the dataset to sample (default: 0.1).

        Returns:
            DataFrame with SMILES strings.
        """
        cached_data = DataCache.load_raw_data("ChEMBL")
        if cached_data is not None:
            logger.info("Loaded ChEMBL from local cache")
            if not hasattr(cached_data, 'shape') and not hasattr(cached_data, 'columns'):
                logger.warning("Cached ChEMBL data has invalid format, re-downloading...")
            else:
                self.data = cached_data
                if sample_frac < 1.0:
                    self.data = self.data.sample(frac=sample_frac).reset_index(drop=True)
                    logger.info(
                        f"Sampled {len(self.data)} molecules from ChEMBL (fraction: {sample_frac})"
                    )
                else:
                    logger.info(f"Loaded {len(self.data)} molecules from ChEMBL.")
                return self.data

        try:
            from tdc.generation import MolGen

            logger.info("Downloading ChEMBL from PyTDC...")
            gen = MolGen(name="ChEMBL")
            self.data = gen.get_data()

            DataCache.save_raw_data(self.data, "ChEMBL")

            if sample_frac < 1.0:
                self.data = self.data.sample(frac=sample_frac).reset_index(drop=True)
                logger.info(
                    f"Sampled {len(self.data)} molecules from ChEMBL (fraction: {sample_frac})"
                )
            else:
                logger.info(f"Loaded {len(self.data)} molecules from ChEMBL.")
            return self.data
        except ImportError:
            raise DataError("TDC package not installed. Run: pip install PyTDC")
        except Exception as e:
            raise DataError(f"Failed to load ChEMBL data: {str(e)}") from e

    def get_data(self) -> Optional[pd.DataFrame]:
        """Get loaded data."""
        return self.data
