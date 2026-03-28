"""DTI (Drug-Target Interaction) trainer module."""

import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

from src.models import GNNDTIPredictor
from src.training.base import Trainer

logger = logging.getLogger(__name__)


class GNNDTITrainer(Trainer):
    """
    Trainer for GNN-based DTI prediction model.

    Features:
    - Optimized training loop with gradient clipping
    - Early stopping with patience
    - Learning rate scheduling
    - Comprehensive metrics tracking
    - RMSE and loss monitoring
    - Mixed-precision training and gradient accumulation for memory efficiency
    """

    def __init__(
        self,
        model: GNNDTIPredictor,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        """
        Initialize DTI trainer.

        Args:
            model: GNNDTIPredictor instance
            device: Training device
            learning_rate: Initial learning rate
            weight_decay: L2 regularization coefficient
        """
        super().__init__(model, device, learning_rate, weight_decay)

        self.criterion = nn.MSELoss()

        # Initialize training history
        self._initialize_history(
            [
                "train_loss",
                "val_loss",
                "val_mae",
                "val_accuracy",
                "learning_rate",
            ]
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
        scaler: GradScaler = None,
        gradient_accumulation_steps: int = 1,
    ) -> float:
        """
        Train for one epoch with optional mixed precision and gradient accumulation.

        Args:
            train_loader: Training data loader
            scaler: GradScaler for mixed precision (optional)
            gradient_accumulation_steps: Number of steps to accumulate gradients (default: 1)

        Returns:
            Average training loss
        """
        if scaler is None:
            scaler = GradScaler(enabled=self.device.type == "cuda")

        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc="Training DTI", leave=False)
        for i, data in enumerate(pbar):
            data = data.to(self.device)

            with autocast(
                device_type=self.device.type, enabled=self.device.type == "cuda"
            ):
                # Forward pass
                predictions = self.model(data, data.prot)
                loss = self.criterion(predictions.squeeze(), data.y)
                loss = loss / gradient_accumulation_steps

            # Backward pass with gradient clipping
            scaler.scale(loss).backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(self.optimizer)
                self._clip_gradients(max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            pbar.set_postfix(
                {"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"}
            )

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model performance with comprehensive metrics.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)

                with autocast(
                    device_type=self.device.type, enabled=self.device.type == "cuda"
                ):
                    predictions = self.model(data, data.prot)
                    loss = self.criterion(predictions.squeeze(), data.y)

                total_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(data.y.cpu().numpy().flatten())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        avg_loss = total_loss / len(val_loader)
        mae = mean_absolute_error(all_targets, all_preds)

        # R² score (accuracy metric for regression)
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "loss": avg_loss,
            "mae": mae,
            "accuracy": r2_score,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = "checkpoints/dti",
        gradient_accumulation_steps: int = 1,
    ) -> Dict[str, Any]:
        """
        Complete training procedure with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            gradient_accumulation_steps: Steps for gradient accumulation

        Returns:
            Dictionary with training results
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        best_model_path = checkpoint_path / "best_model.pt"

        scaler = GradScaler(enabled=self.device.type == "cuda")

        logger.info(f"Starting DTI training for {epochs} epochs...")
        if gradient_accumulation_steps > 1:
            logger.info(
                f"Using gradient accumulation with {gradient_accumulation_steps} steps."
            )

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(
                train_loader, scaler, gradient_accumulation_steps
            )

            # Validate
            val_metrics = self.validate(val_loader)

            # Update learning rate
            lr = self._update_learning_rate(val_metrics["loss"])

            # Record metrics
            self._update_history(
                train_loss=train_loss,
                val_loss=val_metrics["loss"],
                val_mae=val_metrics["mae"],
                val_accuracy=val_metrics["accuracy"],
                learning_rate=lr,
            )

            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"MAE: {val_metrics['mae']:.4f} | "
                f"R²: {val_metrics['accuracy']:.4f}"
            )

            # Early stopping check
            should_continue = self._check_early_stopping(
                val_metrics["loss"],
                is_better=lambda new, best: new < best,
                patience=early_stopping_patience,
            )

            if self.patience_counter == 0:
                # Save best model
                self._save_checkpoint(
                    "gnn_dti",
                    best_model_path,
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        **val_metrics,
                    },
                )

            if not should_continue:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        if best_model_path.exists():
            self._load_checkpoint(best_model_path)

        return {
            "epochs_trained": epoch + 1,
            "best_val_loss": self.best_val_loss,
            "history": self.training_history,
            "checkpoint_path": str(best_model_path),
        }
