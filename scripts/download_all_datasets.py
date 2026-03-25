#!/usr/bin/env python3
"""
Download all datasets required for BioQuest.

Usage:
    python scripts/download_all_datasets.py
"""

import sys
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import TDCDataLoader
from src.data.preparers import PropertyDatasetPreparer


def download_datasets(skip_existing: bool = True):
    """
    Download all supplementary datasets for BioQuest.

    Args:
        skip_existing: Skip download if data already exists in cache
    """
    loader = TDCDataLoader()
    property_preparer = PropertyDatasetPreparer()

    datasets = [
        ("KIBA", lambda: loader.load_dti_data(dataset_name="KIBA")),
        ("BindingDB", lambda: loader.load_dti_data(dataset_name="BindingDB")),
        ("Tox21", lambda: loader.load_tox21_data()),
        ("ChEMBL", lambda: loader.load_chembl_data(sample_frac=0.001)),
        (
            "Lipophilicity",
            lambda: property_preparer.prepare_property_dataset(
                dataset_name="lipophilicity_astrazeneca"
            ),
        ),
    ]

    for name, download_fn in datasets:
        print(f"Downloading {name} dataset...")
        try:
            download_fn()
            print(f"✓ {name} download complete.")
        except Exception as e:
            print(f"✗ Failed to download {name}: {e}")

    print("\nAll downloads complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for BioQuest")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip download if data already exists",
    )
    args = parser.parse_args()
    download_datasets(skip_existing=args.skip_existing)
