"""
Model Utilities and Registry.

Provides model management, checkpointing, and model discovery utilities.
"""

import logging
from typing import List, Dict
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing custom trained models."""

    @staticmethod
    def get_model(model_name: str, **kwargs) -> nn.Module:
        """
        Get model class by name and instantiate.

        Args:
            model_name: Model name (must be in registry)
            **kwargs: Arguments to pass to model constructor

        Returns:
            Instantiated model
        """
        # Import here to avoid circular imports
        from .gnn_dti import GNNDTIPredictor
        from .toxicity import ToxicityClassifier
        from .property import PropertyPredictor

        models = {
            "gnn_dti": GNNDTIPredictor,
            "toxicity": ToxicityClassifier,
            "properties": PropertyPredictor,
        }

        if model_name not in models:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(models.keys())}"
            )

        model_class = models[model_name]
        return model_class(**kwargs)

    @staticmethod
    def list_models() -> List[str]:
        """List all available models."""
        return ["gnn_dti", "toxicity", "properties"]


def create_model_checkpoint(
    model: nn.Module,
    model_name: str,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
) -> None:
    """
    Create and save model checkpoint.

    Args:
        model: Model to save
        model_name: Name of model
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Training metrics
        save_path: Path to save checkpoint
    """
    checkpoint = {
        "model_name": model_name,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> Dict:
    """
    Load model from checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        device: Device to load to

    Returns:
        Checkpoint metadata dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Model loaded from {checkpoint_path}")

    return {
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics"),
        "timestamp": checkpoint.get("timestamp"),
    }
