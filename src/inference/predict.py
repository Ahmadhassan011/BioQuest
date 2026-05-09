"""
Molecule Predictor: Main entry point for inference.

Orchestrates all model predictions for molecules.
"""

import logging
from typing import Dict, List
import numpy as np

from rdkit import Chem

from .dti import DTIPredictor, ModelNotLoadedError as DTIErr
from .toxicity import ToxicityPredictor, ModelNotLoadedError as ToxErr
from .property import PropertyPredictor, ModelNotLoadedError as PropErr
from .vae import VAEGenerator
from ..models.featurization import compute_admet_properties, check_lipinski_rule_of_five

logger = logging.getLogger(__name__)


class MoleculePredictor:
    """
    Integrated predictor using trained neural network models.

    Coordinates DTI, toxicity, and property predictions.

    Raises:
        ModelNotLoadedError: If any model fails to load
    """

    def __init__(
        self,
        protein_sequence: str,
        use_gpu: bool = False,
        models_dir: str = "artifacts/models",
    ):
        """
        Initialize molecule predictor.

        Args:
            protein_sequence: Target protein sequence for DTI predictions
            use_gpu: Whether to use GPU acceleration
            models_dir: Directory containing trained model checkpoints

        Raises:
            ModelNotLoadedError: If models cannot be loaded
        """
        self.protein_sequence = protein_sequence
        self.use_gpu = use_gpu
        self.models_dir = models_dir

        errors = []
        try:
            self._dti = DTIPredictor(protein_sequence, models_dir, use_gpu)
        except DTIErr as e:
            errors.append(f"DTI: {e}")

        try:
            self._toxicity = ToxicityPredictor(models_dir, use_gpu)
        except ToxErr as e:
            errors.append(f"Toxicity: {e}")

        try:
            self._property = PropertyPredictor(models_dir, use_gpu)
        except PropErr as e:
            errors.append(f"Properties: {e}")

        self._vae = VAEGenerator(models_dir, use_gpu)

        if errors:
            raise ModelNotLoadedError(f"Failed to load models: {'; '.join(errors)}")

        logger.info("MoleculePredictor initialized")

    def predict_all_properties(self, smiles: str) -> Dict[str, float]:
        """
        Predict all molecular properties for single molecule.

        Args:
            smiles: Molecule SMILES string

        Returns:
            Dictionary with affinity, toxicity, QED, SA, logp, mw,
            plus ADMET properties (hba, hbd, tpsa, etc.) and Lipinski rules.

        Raises:
            ValueError: If SMILES is invalid
            RuntimeError: If prediction fails
        """
        affinity = self._dti.predict(smiles)
        toxicity = self._toxicity.predict(smiles)
        properties = self._property.predict(smiles)

        mol = Chem.MolFromSmiles(smiles)
        admet = {}
        lipinski = {}
        if mol is not None:
            admet = compute_admet_properties(smiles)
            lipinski = check_lipinski_rule_of_five(admet)

        return {
            "affinity": affinity,
            "toxicity": toxicity,
            "qed": properties["qed"],
            "sa": properties["sa"],
            "logp": properties["logp"],
            "mw": properties["mw"],
            **{k: admet.get(k) for k in ["hba", "hbd", "tpsa", "num_rings",
                "num_aromatic_rings", "num_rotatable_bonds", "num_heavy_atoms",
                "fraction_csp3"]},
            **lipinski,
        }

    def batch_predict(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Predict all properties for batch of molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dictionary with arrays of predictions for each property

        Raises:
            ValueError: If all SMILES are invalid
            RuntimeError: If batch prediction fails
        """
        affinities = self._dti.batch_predict(smiles_list)
        toxicities = self._toxicity.batch_predict(smiles_list)
        properties = self._property.batch_predict(smiles_list)

        return {
            "affinity": affinities,
            "toxicity": toxicities,
            "qed": properties["qed"],
            "sa": properties["sa"],
            "logp": properties["logp"],
            "mw": properties["mw"],
        }

    def score_molecule(self, smiles: str, objective_weights: Dict[str, float]) -> float:
        """
        Calculate weighted composite score.

        Args:
            smiles: Molecule SMILES
            objective_weights: Dictionary of {objective: weight}

        Returns:
            Composite score (0-1 range)

        Raises:
            ValueError: If SMILES is invalid
            RuntimeError: If prediction fails
        """
        properties = self.predict_all_properties(smiles)

        total_weight = sum(objective_weights.values())
        if total_weight == 0:
            raise ValueError("Objective weights sum to zero")

        score = 0.0
        for objective, weight in objective_weights.items():
            prop_value = properties.get(objective)
            if prop_value is None:
                continue

            if objective == "toxicity":
                score += weight * (1.0 - prop_value)
            else:
                score += weight * prop_value

        return score / total_weight

    def get_vae_generator(self) -> VAEGenerator:
        """Get VAE generator for molecule generation."""
        return self._vae


class ModelNotLoadedError(Exception):
    """Raised when any model fails to load."""
    pass