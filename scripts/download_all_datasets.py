import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loaders import TDCDataLoader
from src.data.preparers import (
    PropertyDatasetPreparer,
)  # Correct import for property prediction


def download_datasets():
    """
    Downloads all supplementary datasets mentioned in the project.
    """
    loader = TDCDataLoader()  # Keep this for existing DTI/Tox/ChEMBL downloads
    # Use PropertyDatasetPreparer for property prediction datasets
    property_preparer = PropertyDatasetPreparer()

    print("Downloading KIBA dataset...")
    try:
        loader.load_dti_data(dataset_name="KIBA")
        print("KIBA download complete.")
    except Exception as e:
        print(f"Failed to download KIBA: {e}")

    print("\nDownloading BindingDB dataset...")
    try:
        loader.load_dti_data(dataset_name="BindingDB")
        print("BindingDB download complete.")
    except Exception as e:
        print(f"Failed to download BindingDB: {e}")

    print("\nTox21 dataset...")
    try:
        loader.load_tox21_data()
        print("Tox21 download complete.")
    except Exception as e:
        print(f"Failed to download Tox21: {e}")

    print("\nChEMBL dataset (0.1% sample)...")
    try:
        # Using a very small fraction for quick EDA setup
        loader.load_chembl_data(sample_frac=0.000000000001)
        print("ChEMBL sample download complete.")
    except Exception as e:
        print(f"Failed to download ChEMBL: {e}")

    print("\nDownloading Lipophilicity_ID dataset for Property Prediction...")
    try:
        property_preparer.prepare_property_dataset(
            dataset_name="lipophilicity_astrazeneca"
        )
        print("Lipophilicity_ID download complete.")
    except Exception as e:
        print(f"Failed to download Lipophilicity_ID: {e}")


if __name__ == "__main__":
    download_datasets()
