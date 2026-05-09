"""Data caching and storage management for raw and processed datasets."""

import hashlib
import json
import logging
import pickle
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from infra.exceptions import CacheError

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = "1"

_ROOT_PATH: Optional[Path] = None


def _verify_cache_integrity(filepath: Path) -> bool:
    """Quick integrity check: verify file is non-empty and readable.

    For pickle files (raw cache), checks the pickle STOP opcode at end of file.
    For torch files (processed cache), relies on torch.load's built-in error handling.
    """
    if not filepath.exists() or filepath.stat().st_size == 0:
        return False
    if filepath.suffix == ".pkl":
        try:
            with open(filepath, "rb") as f:
                f.seek(-1, 2)
                if f.read(1) != b".":
                    return False
        except (OSError, EOFError):
            return False
    return True


def _write_version_file(data_dir: Path) -> None:
    """Write cache format version file."""
    (data_dir / "CACHE_VERSION").write_text(CACHE_FORMAT_VERSION)


def _check_version_file(data_dir: Path) -> bool:
    """Check cache format version matches current version."""
    version_file = data_dir / "CACHE_VERSION"
    if not version_file.exists():
        return False
    return version_file.read_text().strip() == CACHE_FORMAT_VERSION


def set_data_root(path: Optional[Path] = None) -> None:
    """Set a custom project root for cache directories.

    Call before any DataCache operation. Pass None to reset to default.
    """
    global _ROOT_PATH
    _ROOT_PATH = path
    _get_project_root.cache_clear()
    _get_data_dir.cache_clear()
    _get_raw_data_dir.cache_clear()
    _get_processed_data_dir.cache_clear()


@lru_cache(maxsize=None)
def _get_project_root() -> Path:
    if _ROOT_PATH is not None:
        return _ROOT_PATH
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def _get_data_dir() -> Path:
    root = _get_project_root() / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


@lru_cache(maxsize=None)
def _get_raw_data_dir() -> Path:
    path = _get_data_dir() / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=None)
def _get_processed_data_dir() -> Path:
    path = _get_data_dir() / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


class DataCache:
    """Manages caching of raw and processed data."""

    @staticmethod
    def get_raw_data_path(dataset_name: str) -> Path:
        """Get path for raw dataset file."""
        return _get_raw_data_dir() / f"{dataset_name}.pkl"

    @staticmethod
    def get_processed_data_path(
        dataset_name: str, preparer_type: str = "dti", suffix: str = ""
    ) -> Path:
        """Get path for processed dataset directory."""
        dir_name = f"{dataset_name}_{preparer_type}"
        if suffix:
            dir_name += f"_{suffix}"
        return _get_processed_data_dir() / dir_name

    @staticmethod
    def save_raw_data(data, dataset_name: str) -> Path:
        """Save raw dataset to local cache."""
        path = DataCache.get_raw_data_path(dataset_name)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        if not _verify_cache_integrity(path):
            path.unlink(missing_ok=True)
            raise CacheError(f"Integrity check failed after saving: {path}")
        logger.info(
            f"Raw data cached: {path} ({path.stat().st_size / 1024 / 1024:.2f} MB)"
        )
        return path

    @staticmethod
    def load_raw_data(dataset_name: str) -> Optional[Any]:
        """Load raw dataset from cache. Returns None if not found."""
        path = DataCache.get_raw_data_path(dataset_name)
        if not path.exists():
            return None
        if not _verify_cache_integrity(path):
            logger.warning(f"Cache file corrupted, removing: {path}")
            path.unlink(missing_ok=True)
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Loaded raw data from cache: {path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache file {path}: {e}")
            path.unlink(missing_ok=True)
            return None

    @staticmethod
    def save_processed_data(
        data_list: Union[list, tuple],
        splits: Dict,
        metadata: Dict,
        dataset_name: str,
        preparer_type: str = "dti",
        suffix: str = "",
    ) -> Path:
        """
        Save processed dataset to local cache.

        Args:
            data_list: List/tuple of processed data objects (or tuple of (features, targets))
            splits: Train/val/test splits
            metadata: Dataset metadata
            dataset_name: Dataset name
            preparer_type: Type of preparer
            suffix: Optional suffix for variants
        """
        data_dir = DataCache.get_processed_data_path(
            dataset_name, preparer_type, suffix
        )
        data_dir.mkdir(parents=True, exist_ok=True)

        _write_version_file(data_dir)

        torch.save(data_list, data_dir / "data_list.pt")

        with open(data_dir / "splits.json", "w") as f:
            splits_serializable = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in splits.items()
            }
            json.dump(splits_serializable, f, indent=2)

        with open(data_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        if isinstance(data_list, tuple):
            num_samples = len(data_list[0])
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
        """Load processed dataset from cache. Returns None if not found."""
        data_dir = DataCache.get_processed_data_path(
            dataset_name, preparer_type, suffix
        )
        if not data_dir.exists():
            return None

        if not _check_version_file(data_dir):
            logger.warning(
                f"Cache format mismatch or missing version file, clearing: {data_dir}"
            )
            shutil.rmtree(data_dir, ignore_errors=True)
            return None

        pt_path = data_dir / "data_list.pt"
        if not _verify_cache_integrity(pt_path):
            logger.warning(f"Cache file corrupted, clearing: {data_dir}")
            shutil.rmtree(data_dir, ignore_errors=True)
            return None

        try:
            data_list = torch.load(pt_path, weights_only=False)

            with open(data_dir / "splits.json", "r") as f:
                splits = json.load(f)
                splits = {
                    k: np.array(v) if isinstance(v, list) else v
                    for k, v in splits.items()
                }

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
        """Check if processed data exists in cache."""
        data_dir = DataCache.get_processed_data_path(
            dataset_name, preparer_type, suffix
        )
        required_files = ["data_list.pt", "splits.json", "metadata.json", "CACHE_VERSION"]
        return all((data_dir / f).exists() for f in required_files) and _check_version_file(data_dir)

    @staticmethod
    def clear_cache(
        dataset_name: Optional[str] = None, processed_only: bool = False
    ) -> None:
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

            for processed_dir in _get_processed_data_dir().glob(f"{dataset_name}_*"):
                shutil.rmtree(processed_dir)
                logger.info(f"Deleted processed data cache: {processed_dir}")
        else:
            if not processed_only:
                for raw_file in _get_raw_data_dir().glob("*.pkl"):
                    raw_file.unlink()
                logger.info("Cleared all raw data cache")

            for processed_dir in _get_processed_data_dir().glob("*"):
                shutil.rmtree(processed_dir)
            logger.info("Cleared all processed data cache")

    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "data_dir": str(_get_data_dir()),
            "raw_data_size_mb": 0,
            "processed_data_size_mb": 0,
            "raw_datasets": [],
            "processed_datasets": [],
        }

        for raw_file in _get_raw_data_dir().glob("*.pkl"):
            size_mb = raw_file.stat().st_size / 1024 / 1024
            stats["raw_data_size_mb"] += size_mb
            stats["raw_datasets"].append(
                {"name": raw_file.stem, "size_mb": round(size_mb, 2)}
            )

        for processed_dir in _get_processed_data_dir().glob("*"):
            if processed_dir.is_dir():
                size_mb = (
                    sum(f.stat().st_size for f in processed_dir.glob("*"))
                    / 1024
                    / 1024
                )
                stats["processed_data_size_mb"] += size_mb

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
                print(f"  {ds['name']:<20} {ds['size_mb']:>10.2f} MB")
            print(f"  {'Total':<20} {stats['raw_data_size_mb']:>10.2f} MB")
        else:
            print("  (No raw data cached)")

        print("\nPROCESSED DATA:")
        if stats["processed_datasets"]:
            for ds in stats["processed_datasets"]:
                print(f"  {ds['name']:<35} {ds['size_mb']:>10.2f} MB")
                if "num_samples" in ds:
                    print(f"    Samples: {ds.get('num_samples', 'N/A')}")
            print(f"  {'Total':<35} {stats['processed_data_size_mb']:>10.2f} MB")
        else:
            print("  (No processed data cached)")

        print(f"\nTOTAL: {stats['total_size_mb']:.2f} MB")
        print("=" * 70 + "\n")
