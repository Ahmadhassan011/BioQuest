"""
Property inference.

Handles property prediction (QED, SA, LogP, MW) for molecules.
"""

import logging
from typing import List, Dict
import numpy as np
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelNotLoadedError(Exception):
    """Raised when property model is not loaded."""
    pass


class PropertyPredictor:
    """Property inference for multi-task property prediction."""

    def __init__(self, models_dir: str = "artifacts/models", use_gpu: bool = False):
        """
        Initialize property predictor.

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
            raise ModelNotLoadedError(f"Property model not found in {models_dir}/properties/")

    def _load_models(self, models_dir: str) -> None:
        """Load property model and initialize featurizer."""
        from src.models.property import PropertyPredictor as PropertyModel
        from src.models.featurization import MolecularFeaturizer

        self._featurizer = MolecularFeaturizer()

        model_path = Path(models_dir) / "properties" / "best_model.pt"
        if not model_path.exists():
            logger.error(f"Property model not found at {model_path}")
            return

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            arch_config = checkpoint.get("model_config", {})

            self._model = PropertyModel(
                input_dim=arch_config.get("input_dim", 264),
                shared_hidden_dim=arch_config.get("shared_hidden_dim", 256),
                task_hidden_dim=arch_config.get("task_hidden_dim", 128),
            )
            self._model.to(self.device)

            state = checkpoint.get("model_state_dict", checkpoint)
            missing, unexpected = self._model.load_state_dict(state, strict=False)
            if missing or unexpected:
                logger.warning(f"Property model state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
            self._model.eval()
            logger.info(f"Property model loaded from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load property model: {e}")
            self._model = None

    def predict(self, smiles: str) -> Dict[str, float]:
        """
        Predict properties for single molecule.

        Args:
            smiles: Molecule SMILES

        Returns:
            Dictionary with QED, SA, LogP, MW

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
                predictions = self._model(mol_features)

            return {
                "qed": float(predictions["qed"].cpu().numpy()[0, 0]),
                "sa": float(predictions["sa"].cpu().numpy()[0, 0]),
                "logp": float(predictions["logp"].cpu().numpy()[0, 0]),
                "mw": float(predictions["mw"].cpu().numpy()[0, 0]),
            }

        except Exception as e:
            if "featurize" in str(e).lower() or "smiles" in str(e).lower():
                raise ValueError(f"Invalid SMILES: {smiles}") from e
            raise RuntimeError(f"Property prediction failed: {e}") from e

    def batch_predict(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Predict properties for batch of molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dictionary with arrays of predictions

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
                predictions = self._model(mol_features_batch)

            return {
                "qed": predictions["qed"].cpu().numpy().flatten(),
                "sa": predictions["sa"].cpu().numpy().flatten(),
                "logp": predictions["logp"].cpu().numpy().flatten(),
                "mw": predictions["mw"].cpu().numpy().flatten(),
            }

        except Exception as e:
            raise RuntimeError(f"Batch property prediction failed: {e}") from e