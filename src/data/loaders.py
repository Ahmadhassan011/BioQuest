"""
Data Module: Protein sequence handling, seed molecule management, dataset loading.

This module provides utilities to:
- Load and validate protein sequences
- Manage seed molecule SMILES strings
- Interface with PyTDC for drug-target interaction data
- Cache datasets locally in project/data/ directory
- Prepare inputs for molecule generation and prediction

Datasets Used:
- DAVIS: DTI data (4485 proteins, 68 drugs, 30056 interactions)
  - Sequences longer than 1000 AA truncated
- Tox21: Toxicity data (7831 compounds, 12 assays)
  - Inverted labels for scoring (lower is better)
- ChEMBL: Molecule source for VAE pretraining (millions of molecules)
  - Molecules canonicalized and filtered for drug-likeness

DATA STORAGE:
  Local caching in: <project_root>/data/
  ├── raw/          - Downloaded datasets from PyTDC
  └── processed/    - Featurized, normalized datasets
"""

import logging
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
# Lazy-import TDC modules inside functions to avoid import-time failures
# when PyTDC is not installed or API differs across versions.

from src.data.storage import DataCache

# Configure logging
logger = logging.getLogger(__name__)


class ProteinDataHandler:
    """Handles protein sequence loading, validation, and storage."""

    def __init__(self, max_sequence_length: int = 2000):
        """
        Initialize protein data handler.

        Args:
            max_sequence_length: Maximum allowed protein sequence length
        """
        self.max_sequence_length = max_sequence_length
        self.valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        self.amino_acids = sorted(list(self.valid_amino_acids))
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}

    def validate_protein_sequence(self, sequence: str) -> str:
        """
        Validate and potentially truncate protein sequence format.

        Args:
            sequence: Protein sequence string

        Returns:
            The validated and potentially truncated sequence string.
            Raises ValueError if sequence contains invalid amino acids or is not a string.
        """
        if not isinstance(sequence, str):
            raise ValueError("Protein sequence must be a string")

        original_length = len(sequence)
        if original_length > self.max_sequence_length:
            logger.warning(
                f"Sequence length {original_length} exceeds max {self.max_sequence_length}. Truncating sequence."
            )
            sequence = sequence[: self.max_sequence_length]

        sequence = sequence.upper()
        if not all(aa in self.valid_amino_acids for aa in sequence):
            raise ValueError("Sequence contains invalid amino acids")

        return sequence

    def prepare_protein_indices(self, sequence: str) -> np.ndarray:
        """
        Convert protein sequence to an array of indices.

        Args:
            sequence: Protein sequence string

        Returns:
            A numpy array of amino acid indices.
        """
        validated_sequence = self.validate_protein_sequence(sequence)

        indices = [self.aa_to_idx.get(aa, 0) for aa in validated_sequence]

        return np.array(indices, dtype=np.int64)


class MoleculeSeedHandler:
    """Manages seed molecule SMILES strings."""

    def __init__(self):
        """Initialize molecule seed handler."""
        self.seeds: List[str] = []

    def add_seed(self, smiles: str) -> None:
        """
        Add a seed molecule SMILES string.

        Args:
            smiles: SMILES string representation
        """
        if isinstance(smiles, str) and len(smiles) > 0:
            self.seeds.append(smiles)
            logger.info(f"Added seed molecule: {smiles}")

    def add_seeds_from_list(self, smiles_list: List[str]) -> None:
        """
        Add multiple seed molecules.

        Args:
            smiles_list: List of SMILES strings
        """
        for smiles in smiles_list:
            self.add_seed(smiles)

    def get_seeds(self) -> List[str]:
        """Get all seed molecules."""
        return self.seeds.copy()

    def clear_seeds(self) -> None:
        """Clear all seeds."""
        self.seeds = []


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
        # Check local cache first
        cached_data = DataCache.load_raw_data(dataset_name)
        if cached_data is not None:
            logger.info(f"Loaded {dataset_name} from local cache")
            self.data = cached_data
            return self.data

        try:
            from tdc.multi_pred import DTI

            logger.info(f"Downloading {dataset_name} from PyTDC...")
            dti = DTI(name=dataset_name)
            self.data = dti.get_data()

            # Save to local cache
            DataCache.save_raw_data(self.data, dataset_name)

            logger.info(f"Loaded {len(self.data)} DTI interactions from {dataset_name}")
            return self.data
        except Exception as e:
            logger.error(f"Failed to load DTI data: {str(e)}")
            raise

    def load_tox21_data(self, assay: Optional[str] = None) -> pd.DataFrame:
        """
        Load Tox21 dataset from PyTDC (with local caching).

        Args:
            assay: Specific Tox21 assay to load (e.g., 'NR-AR'). If None, loads all.

        Returns:
            DataFrame with toxicity data.
        """
        # Check local cache first
        if assay is None:
            cached_data = DataCache.load_raw_data("Tox21")
            if cached_data is not None:
                logger.info("Loaded Tox21 from local cache")
                self.data = cached_data
                return self.data
        else:
            cached_data = DataCache.load_raw_data(f"Tox21_{assay}")
            if cached_data is not None:
                logger.info(f"Loaded Tox21_{assay} from local cache")
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
        except Exception as e:
            logger.error(f"Failed to load Tox21 data: {str(e)}")
            raise

    def load_chembl_data(self, sample_frac: float = 0.1) -> pd.DataFrame:
        """
        Load ChEMBL dataset for molecule generation pretraining (with local caching).

        Args:
            sample_frac: Fraction of the dataset to sample (default: 0.1).

        Returns:
            DataFrame with SMILES strings.
        """
        # Check local cache first
        cached_data = DataCache.load_raw_data("ChEMBL")
        if cached_data is not None:
            logger.info("Loaded ChEMBL from local cache")
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

            # Save to local cache
            DataCache.save_raw_data(self.data, "ChEMBL")

            if sample_frac < 1.0:
                self.data = self.data.sample(frac=sample_frac).reset_index(drop=True)
                logger.info(
                    f"Sampled {len(self.data)} molecules from ChEMBL (fraction: {sample_frac})"
                )
            else:
                logger.info(f"Loaded {len(self.data)} molecules from ChEMBL.")
            return self.data
        except Exception as e:
            logger.error(f"Failed to load ChEMBL data: {str(e)}")
            raise

    def get_data(self) -> Optional[pd.DataFrame]:
        """Get loaded data."""
        return self.data


class ObjectiveHandler:
    """Manages multi-objective optimization targets."""

    def __init__(self):
        """Initialize objective handler."""
        self.objectives: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}

    def add_objective(
        self, name: str, weight: float = 1.0, target: Optional[float] = None
    ) -> None:
        """
        Add optimization objective.

        Args:
            name: Objective name (e.g., "affinity", "toxicity", "qed", "sa")
            weight: Weight for multi-objective optimization (0-1 range recommended)
            target: Target value for objective (optional)
        """
        if weight < 0:
            raise ValueError("Objective weight must be non-negative")

        self.objectives[name] = target
        self.weights[name] = weight
        logger.info(f"Added objective '{name}' with weight {weight}")

    def get_objectives(self) -> Dict[str, float]:
        """Get all objectives and targets."""
        return self.objectives.copy()

    def get_weights(self) -> Dict[str, float]:
        """Get objective weights."""
        return self.weights.copy()

    def normalize_weights(self) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0.

        Returns:
            Dictionary of normalized weights
        """
        total = sum(self.weights.values())
        if total == 0:
            return {k: 1.0 / len(self.weights) for k in self.weights}
        return {k: v / total for k, v in self.weights.items()}


class BioQuestDataset:
    """Main dataset manager combining proteins, seeds, objectives, and DTI data."""

    def __init__(self):
        """Initialize BioQuest dataset."""
        self.protein_handler = ProteinDataHandler()
        self.seed_handler = MoleculeSeedHandler()
        self.objective_handler = ObjectiveHandler()
        self.tdc_loader = TDCDataLoader()
        self.protein_sequence: Optional[str] = None

    def set_protein_sequence(self, sequence: str) -> bool:
        """
        Set the target protein sequence.

        Args:
            sequence: Protein sequence string

        Returns:
            True if valid and set, False otherwise
        """
        if self.protein_handler.validate_protein_sequence(sequence):
            self.protein_sequence = sequence.upper()
            logger.info(f"Set protein sequence (length: {len(sequence)})")
            return True
        return False

    def add_seeds(self, smiles_list: List[str]) -> None:
        """Add seed molecules."""
        self.seed_handler.add_seeds_from_list(smiles_list)

    def add_objective(
        self, name: str, weight: float = 1.0, target: Optional[float] = None
    ) -> None:
        """
        Add optimization objective.

        Args:
            name: Objective name (e.g., "affinity", "toxicity", "qed", "sa")
            weight: Weight for multi-objective optimization (0-1 range recommended)
            target: Target value for objective (optional)
        """
        self.objective_handler.add_objective(name, weight=weight, target=target)

    def get_protein_embedding(self) -> Optional[np.ndarray]:
        """Get protein sequence embedding."""
        if self.protein_sequence:
            return self.protein_handler.prepare_protein_indices(self.protein_sequence)
        return None

    def get_seeds(self) -> List[str]:
        """Get seed molecules."""
        return self.seed_handler.get_seeds()

    def get_objectives(self) -> Dict[str, float]:
        """Get objectives."""
        return self.objective_handler.get_objectives()

    def get_objective_weights(self) -> Dict[str, float]:
        """Get normalized objective weights."""
        return self.objective_handler.normalize_weights()

    def summary(self) -> Dict:
        """
        Get dataset summary.

        Returns:
            Dictionary with configuration summary
        """
        return {
            "protein_length": len(self.protein_sequence)
            if self.protein_sequence
            else 0,
            "num_seeds": len(self.seed_handler.get_seeds()),
            "objectives": self.objective_handler.get_objectives(),
            "weights": self.objective_handler.normalize_weights(),
        }


# Module-level utilities
def create_dataset(
    protein_sequence: str,
    seed_smiles: List[str],
    objectives: Dict[str, float],
) -> BioQuestDataset:
    """
    Factory function to create and configure BioQuest dataset.

    Args:
        protein_sequence: Target protein sequence
        seed_smiles: List of seed molecule SMILES strings
        objectives: Dictionary of {objective_name: weight}

    Returns:
        Configured BioQuestDataset instance
    """
    dataset = BioQuestDataset()

    if not dataset.set_protein_sequence(protein_sequence):
        raise ValueError("Invalid protein sequence")

    dataset.add_seeds(seed_smiles)

    for obj_name, weight in objectives.items():
        dataset.add_objective(obj_name, weight)

    logger.info("Dataset created successfully")
    return dataset
