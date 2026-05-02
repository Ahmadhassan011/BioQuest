"""
Data Caching & Storage Management Module.

Manages local storage of:
- Raw datasets (downloaded from PyTDC)
- Processed datasets (featurized, normalized)
- Training metadata

All data stored in: <project_root>/data/ directory
"""

import logging
import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Project data directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


class DataCache:
    """Manages caching of raw and processed data."""

    @staticmethod
    def get_raw_data_path(dataset_name: str) -> Path:
        """
        Get path for raw dataset.

        Args:
            dataset_name: Dataset name (e.g., 'DAVIS', 'Tox21', 'ChEMBL')

        Returns:
            Path to raw data file
        """
        return RAW_DATA_DIR / f"{dataset_name}.pkl"

    @staticmethod
    def get_processed_data_path(
        dataset_name: str, preparer_type: str = "dti", suffix: str = ""
    ) -> Path:
        """
        Get path for processed dataset.

        Args:
            dataset_name: Dataset name (e.g., 'DAVIS', 'Tox21')
            preparer_type: Type of preparer (dti, toxicity, vae)
            suffix: Optional suffix for variants

        Returns:
            Path to processed data directory
        """
        dir_name = f"{dataset_name}_{preparer_type}"
        if suffix:
            dir_name += f"_{suffix}"
        return PROCESSED_DATA_DIR / dir_name

    @staticmethod
    def save_raw_data(data, dataset_name: str) -> Path:
        """
        Save raw dataset locally.

        Args:
            data: Dataset (pandas DataFrame or similar)
            dataset_name: Dataset name

        Returns:
            Path where data was saved
        """
        path = DataCache.get_raw_data_path(dataset_name)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(
            f"Raw data cached: {path} ({path.stat().st_size / 1024 / 1024:.2f} MB)"
        )
        return path

    @staticmethod
    def load_raw_data(dataset_name: str) -> Optional[Any]:
        """
        Load raw dataset from cache.

        Args:
            dataset_name: Dataset name

        Returns:
            Dataset or None if not found
        """
        path = DataCache.get_raw_data_path(dataset_name)
        if not path.exists():
            return None

        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Loaded raw data from cache: {path}")
        return data

    @staticmethod
    def save_processed_data(
        data_list: list,
        splits: Dict,
        metadata: Dict,
        dataset_name: str,
        preparer_type: str = "dti",
        suffix: str = "",
    ) -> Path:
        """
        Save processed dataset locally.

        Args:
            data_list: List of processed data objects
            splits: Train/val/test splits
            metadata: Dataset metadata
            dataset_name: Dataset name
            preparer_type: Type of preparer
            suffix: Optional suffix

        Returns:
            Path to processed data directory
        """
        data_dir = DataCache.get_processed_data_path(
            dataset_name, preparer_type, suffix
        )
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save data list
        torch.save(data_list, data_dir / "data_list.pt")

        # Save splits
        with open(data_dir / "splits.json", "w") as f:
            splits_serializable = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in splits.items()
            }
            json.dump(splits_serializable, f, indent=2)

        # Save metadata
        with open(data_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save cache info
        # Handle both lists (DTI) and tuples (Properties: (features, targets))
        if isinstance(data_list, tuple):
            num_samples = len(data_list[0])  # First element contains the data
        else:
            num_samples = len(data_list)
        cache_info = {
            "dataset_name": dataset_name,
            "preparer_type": preparer_type,
            "num_samples": num_samples,
            "train_size": len(splits.get("train", [])),
            "val_size": len(splits.get("val", [])),
            "test_size": len(splits.get("test", [])),
        }
        with open(data_dir / "cache_info.json", "w") as f:
            json.dump(cache_info, f, indent=2)

        total_size = sum(f.stat().st_size for f in data_dir.glob("*") if f.is_file())
        logger.info(
            f"Processed data cached: {data_dir} "
            f"({total_size / 1024 / 1024:.2f} MB, {num_samples} samples)"
        )
        return data_dir

    @staticmethod
    def load_processed_data(
        dataset_name: str, preparer_type: str = "dti", suffix: str = ""
    ) -> Optional[tuple]:
        """
        Load processed dataset from cache.

        Args:
            dataset_name: Dataset name
            preparer_type: Type of preparer
            suffix: Optional suffix

        Returns:
            Tuple of (data_list, splits, metadata) or None if not found
        """
        data_dir = DataCache.get_processed_data_path(
            dataset_name, preparer_type, suffix
        )

        if not data_dir.exists():
            return None

        try:
            # Load data list
            data_list = torch.load(data_dir / "data_list.pt", weights_only=False)

            # Load splits
            with open(data_dir / "splits.json", "r") as f:
                splits = json.load(f)
                splits = {
                    k: np.array(v) if isinstance(v, list) else v
                    for k, v in splits.items()
                }

            # Load metadata
            with open(data_dir / "metadata.json", "r") as f:
                metadata = json.load(f)

            logger.info(f"Loaded processed data from cache: {data_dir}")
            return data_list, splits, metadata
        except Exception as e:
            logger.warning(f"Failed to load processed data from cache: {e}")
            return None

    @staticmethod
    def has_processed_data(
        dataset_name: str, preparer_type: str = "dti", suffix: str = ""
    ) -> bool:
        """
        Check if processed data exists in cache.

        Args:
            dataset_name: Dataset name
            preparer_type: Type of preparer
            suffix: Optional suffix

        Returns:
            True if cached data exists
        """
        data_dir = DataCache.get_processed_data_path(
            dataset_name, preparer_type, suffix
        )
        required_files = ["data_list.pt", "splits.json", "metadata.json"]
        return all((data_dir / f).exists() for f in required_files)

    @staticmethod
    def clear_cache(dataset_name: str = None, processed_only: bool = False) -> None:
        """
        Clear cached data.

        Args:
            dataset_name: Specific dataset to clear (None = all)
            processed_only: Only clear processed data (keep raw)
        """
        if dataset_name:
            if not processed_only:
                raw_path = DataCache.get_raw_data_path(dataset_name)
                if raw_path.exists():
                    raw_path.unlink()
                    logger.info(f"Deleted raw data cache: {raw_path}")

            # Delete all processed variants
            for processed_dir in PROCESSED_DATA_DIR.glob(f"{dataset_name}_*"):
                import shutil

                shutil.rmtree(processed_dir)
                logger.info(f"Deleted processed data cache: {processed_dir}")
        else:
            if not processed_only:
                for raw_file in RAW_DATA_DIR.glob("*.pkl"):
                    raw_file.unlink()
                logger.info("Cleared all raw data cache")

            import shutil

            for processed_dir in PROCESSED_DATA_DIR.glob("*"):
                shutil.rmtree(processed_dir)
            logger.info("Cleared all processed data cache")

    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache info
        """
        stats = {
            "data_dir": str(DATA_DIR),
            "raw_data_size_mb": 0,
            "processed_data_size_mb": 0,
            "raw_datasets": [],
            "processed_datasets": [],
        }

        # Raw data stats
        for raw_file in RAW_DATA_DIR.glob("*.pkl"):
            size_mb = raw_file.stat().st_size / 1024 / 1024
            stats["raw_data_size_mb"] += size_mb
            stats["raw_datasets"].append(
                {"name": raw_file.stem, "size_mb": round(size_mb, 2)}
            )

        # Processed data stats
        for processed_dir in PROCESSED_DATA_DIR.glob("*"):
            if processed_dir.is_dir():
                size_mb = (
                    sum(f.stat().st_size for f in processed_dir.glob("*")) / 1024 / 1024
                )
                stats["processed_data_size_mb"] += size_mb

                # Read cache info
                cache_info_path = processed_dir / "cache_info.json"
                info = {}
                if cache_info_path.exists():
                    with open(cache_info_path, "r") as f:
                        info = json.load(f)

                stats["processed_datasets"].append(
                    {"name": processed_dir.name, "size_mb": round(size_mb, 2), **info}
                )

        stats["raw_data_size_mb"] = round(stats["raw_data_size_mb"], 2)
        stats["processed_data_size_mb"] = round(stats["processed_data_size_mb"], 2)
        stats["total_size_mb"] = round(
            stats["raw_data_size_mb"] + stats["processed_data_size_mb"], 2
        )

        return stats

    @staticmethod
    def print_cache_summary() -> None:
        """Print cache statistics to console."""
        stats = DataCache.get_cache_stats()

        print("\n" + "=" * 70)
        print("DATA CACHE SUMMARY")
        print("=" * 70)
        print(f"\nCache Directory: {stats['data_dir']}\n")

        print("RAW DATA:")
        if stats["raw_datasets"]:
            for ds in stats["raw_datasets"]:
                print(f"  • {ds['name']:<20} {ds['size_mb']:>10.2f} MB")
            print(f"  {'Total':<20} {stats['raw_data_size_mb']:>10.2f} MB")
        else:
            print("  (No raw data cached)")

        print("\nPROCESSED DATA:")
        if stats["processed_datasets"]:
            for ds in stats["processed_datasets"]:
                print(f"  • {ds['name']:<35} {ds['size_mb']:>10.2f} MB")
                if "num_samples" in ds:
                    print(f"    Samples: {ds.get('num_samples', 'N/A')}")
            print(f"  {'Total':<35} {stats['processed_data_size_mb']:>10.2f} MB")
        else:
            print("  (No processed data cached)")

        print(f"\nTOTAL: {stats['total_size_mb']:.2f} MB")
        print("=" * 70 + "\n")
