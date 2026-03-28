"""Toxicity classifier trainer module."""

import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

try:
    from sklearn.metrics import matthews_corrcoef
except ImportError:
    from sklearn.metrics.cluster import supervised as _mcc_module

    def matthews_corrcoef(y_true, y_pred, *, sample_weight=None, zero_division="warn"):
        return _mcc_module.matthews_corrcoef(
            y_true, y_pred, sample_weight=sample_weight
        )


from src.models import ToxicityClassifier
from src.training.base import Trainer

logger = logging.getLogger(__name__)


class ToxicityClassifierTrainer(Trainer):
    """
    Trainer for toxicity classification model.

    Features:
    - Binary classification with BCE loss
    - ROC-AUC monitoring
    - Early stopping based on validation metrics
    - Gradient clipping for stability
    """

    def __init__(
        self,
        model: ToxicityClassifier,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        """
        Initialize toxicity trainer.

        Args:
            model: ToxicityClassifier instance
            device: Training device
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
        """
        super().__init__(model, device, learning_rate, weight_decay)

        self.criterion = nn.BCELoss()
        self.best_val_auc = 0.0

        # Initialize training history
        self._initialize_history(
            [
                "train_loss",
                "val_loss",
                "val_auc",
                "val_f1",
                "val_mcc",
                "learning_rate",
            ]
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc="Training Toxicity", leave=False)
        for features, labels in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.criterion(predictions, labels)

            loss.backward()
            self._clip_gradients(max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(train_loader)

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
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                predictions = self.model(features)
                loss = self.criterion(predictions, labels)

                total_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        avg_loss = total_loss / len(val_loader)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        binary_preds = (all_preds > 0.5).astype(int)

        # Calculate ROC-AUC
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
            auc = 0.0

        # Calculate F1 and MCC (omit Accuracy, Precision, Recall as redundant)
        try:
            f1 = f1_score(all_labels, binary_preds, zero_division=0)
            mcc = matthews_corrcoef(all_labels, binary_preds)
        except Exception as e:
            logger.warning(f"Could not calculate classification metrics: {e}")
            f1 = mcc = 0.0

        return {
            "loss": avg_loss,
            "auc": auc,
            "f1": f1,
            "mcc": mcc,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = "checkpoints/toxicity",
    ) -> Dict[str, Any]:
        """
        Train toxicity model with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training results dictionary
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        best_model_path = checkpoint_path / "best_model.pt"

        logger.info(f"Starting toxicity training for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            lr = self._update_learning_rate(val_metrics["loss"])

            self._update_history(
                train_loss=train_loss,
                val_loss=val_metrics["loss"],
                val_auc=val_metrics["auc"],
                val_f1=val_metrics["f1"],
                val_mcc=val_metrics["mcc"],
                learning_rate=lr,
            )

            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"AUC: {val_metrics['auc']:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | "
                f"MCC: {val_metrics['mcc']:.4f}"
            )

            # Early stopping using AUC (higher is better)
            should_continue = self._check_early_stopping(
                val_metrics["auc"],
                is_better=lambda new, best: new > best,
                patience=early_stopping_patience,
            )

            if self.patience_counter == 0:
                self._save_checkpoint(
                    "toxicity",
                    best_model_path,
                    val_metrics,
                )

            if not should_continue:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        if best_model_path.exists():
            self._load_checkpoint(best_model_path)

        return {
            "epochs_trained": epoch + 1,
            "best_val_auc": self.best_val_auc,
            "history": self.training_history,
            "checkpoint_path": str(best_model_path),
        }
