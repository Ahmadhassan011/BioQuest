"""
Toxicity inference.

Handles toxicity prediction for molecules.
"""

import logging
from typing import List
import numpy as np
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelNotLoadedError(Exception):
    """Raised when toxicity model is not loaded."""
    pass


class ToxicityPredictor:
    """Toxicity inference for molecule toxicity prediction."""

    def __init__(self, models_dir: str = "artifacts/models", use_gpu: bool = False):
        """
        Initialize toxicity predictor.

        Args:
            models_dir: Directory containing trained models
            use_gpu: Whether to use GPU

        Raises:
            ModelNotLoadedError: If model cannot be loaded
        """
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self._model = None
        self._featurizer = None
        self._load_models(models_dir)

        if self._model is None:
            raise ModelNotLoadedError(f"Toxicity model not found in {models_dir}/toxicity/")

    def _load_models(self, models_dir: str) -> None:
        """Load toxicity model and initialize featurizer."""
        from src.models.toxicity import ToxicityClassifier
        from src.models.featurization import MolecularFeaturizer

        self._featurizer = MolecularFeaturizer()

        model_path = Path(models_dir) / "toxicity" / "best_model.pt"
        if not model_path.exists():
            logger.error(f"Toxicity model not found at {model_path}")
            return

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            arch_config = checkpoint.get("model_config", {})

            self._model = ToxicityClassifier(
                input_dim=arch_config.get("input_dim", 264),
                hidden_dims=arch_config.get("hidden_dims", [512, 256, 128, 64]),
                dropout=arch_config.get("dropout", 0.3),
            )
            self._model.to(self.device)

            state = checkpoint.get("model_state_dict", checkpoint)
            missing, unexpected = self._model.load_state_dict(state, strict=False)
            if missing or unexpected:
                logger.warning(f"Toxicity model state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
            self._model.eval()
            logger.info(f"Toxicity model loaded from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load toxicity model: {e}")
            self._model = None

    def predict(self, smiles: str) -> float:
        """
        Predict toxicity for single molecule.

        Args:
            smiles: Molecule SMILES

        Returns:
            Predicted toxicity (0-1, where 1 is most toxic)

        Raises:
            ValueError: If SMILES is invalid
            RuntimeError: If prediction fails
        """
        try:
            mol_features = (
                torch.from_numpy(self._featurizer.featurize_molecule(smiles))
                .float()
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                toxicity = self._model(mol_features)

            return float(toxicity.cpu().numpy()[0, 0])

        except Exception as e:
            if "featurize" in str(e).lower() or "smiles" in str(e).lower():
                raise ValueError(f"Invalid SMILES: {smiles}") from e
            raise RuntimeError(f"Toxicity prediction failed: {e}") from e

    def batch_predict(self, smiles_list: List[str]) -> np.ndarray:
        """
        Predict toxicities for batch of molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Array of predicted toxicities

        Raises:
            RuntimeError: If batch prediction fails
        """
        try:
            mol_features_batch = (
                torch.from_numpy(self._featurizer.batch_featurize_molecules(smiles_list))
                .float()
                .to(self.device)
            )

            with torch.no_grad():
                toxicities = self._model(mol_features_batch)

            return toxicities.cpu().numpy().flatten()

        except Exception as e:
            raise RuntimeError(f"Batch toxicity prediction failed: {e}") from e