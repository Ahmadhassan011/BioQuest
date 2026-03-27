"""
Core trainer base class with shared training infrastructure.

Provides:
- Optimizer and learning rate scheduler management
- Early stopping with patience
- Gradient clipping
- Checkpoint save/load
- Training history tracking
"""

import logging
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class Trainer:
    """
    Base trainer class with common training infrastructure.

    All model-specific trainers inherit from this class to leverage:
    - Standard optimizer and scheduler setup
    - Training loop template
    - Model checkpointing
    - Metrics tracking
    - Early stopping logic

    All trainer subclasses must implement:
    - train_epoch(loader)
    - validate(loader)
    - fit(train_loader, val_loader, ...)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        """
        Initialize base trainer.

        Args:
            model: Neural network model to train
            device: Training device (cuda/cpu)
            learning_rate: Initial learning rate
            weight_decay: L2 regularization coefficient
        """
        self.model = model
        self.device = device
        self.model.to(device)

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history = {}

    def _initialize_history(self, keys: list):
        """Initialize training history with given keys."""
        self.training_history = {key: [] for key in keys}

    def _update_history(self, **kwargs):
        """Update training history with new values."""
        for key, value in kwargs.items():
            if key in self.training_history:
                self.training_history[key].append(value)

    def _save_checkpoint(
        self,
        model_name: str,
        checkpoint_path: Path,
        metadata: Dict[str, Any] = None,
    ):
        """
        Save model checkpoint.

        Args:
            model_name: Name identifier for model
            checkpoint_path: Path to save checkpoint
            metadata: Additional metadata to save, including 'epoch'
        """
        from src.models.registry import create_model_checkpoint

        metadata = metadata or {}
        epoch = metadata.pop("epoch", -1)
        metrics = metadata

        create_model_checkpoint(
            model=self.model,
            model_name=model_name,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics=metrics,
            save_path=str(checkpoint_path),
        )
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        from src.models.registry import load_model_checkpoint

        load_model_checkpoint(self.model, str(checkpoint_path), self.device)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def _clip_gradients(self, max_norm: float = 1.0):
        """Clip model gradients to prevent explosion."""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

    def _update_learning_rate(self, metric: float):
        """Update learning rate based on metric (e.g., validation loss)."""
        self.scheduler.step(metric)
        return self.optimizer.param_groups[0]["lr"]

    def _check_early_stopping(
        self,
        val_metric: float,
        is_better: callable = lambda new, best: new < best,
        patience: int = 10,
    ) -> bool:
        """
        Check if early stopping condition is met.

        Args:
            val_metric: Current validation metric
            is_better: Function to determine if new metric is better
            patience: Patience for early stopping

        Returns:
            True if should continue training, False if should stop
        """
        if is_better(val_metric, self.best_val_loss):
            self.best_val_loss = val_metric
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return self.patience_counter < patience
