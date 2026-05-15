"""Toxicity classifier trainer module."""

import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import matthews_corrcoef

from ..models import ToxicityClassifier
from .base import Trainer

logger = logging.getLogger(__name__)


class ToxicityClassifierTrainer(Trainer):
    """
    Trainer for toxicity classification model.

    Features:
    - Binary classification with weighted BCE loss (handles class imbalance)
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
        pos_weight: float = None,
    ):
        """
        Initialize toxicity trainer.

        Args:
            model: ToxicityClassifier instance
            device: Training device
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            pos_weight: Positive class weight for imbalance (default: auto-computed)
        """
        super().__init__(model, device, learning_rate, weight_decay)

        self.pos_weight = pos_weight
        self.criterion = None  # Set after computing pos_weight
        self.best_val_auc = 0.0
        self.best_val_loss = -1.0  # AUC is in [0,1], higher is better

        # Initialize training history
        self._initialize_history(
            [
                "train_loss",
                "val_loss",
                "val_auc",
                "val_pr_auc",
                "val_f1",
                "val_mcc",
                "learning_rate",
            ]
        )

    def _init_criterion(self, train_loader: DataLoader):
        """Compute pos_weight from training data and initialize weighted loss."""
        if self.pos_weight is not None or self.criterion is not None:
            return

        labels = []
        for features, target in train_loader:
            labels.append(target)
        all_labels = torch.cat(labels)

        n_neg = (all_labels == 0).sum().item()
        n_pos = (all_labels == 1).sum().item()

        if n_pos > 0:
            self.pos_weight = n_neg / n_pos
            logger.info(
                f"Class imbalance detected: neg={n_neg}, pos={n_pos}, "
                f"pos_weight={self.pos_weight:.2f}"
            )
        else:
            self.pos_weight = 1.0

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]).to(self.device))

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self._init_criterion(train_loader)
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc="Training Toxicity", leave=False)
        for features, labels in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            logits = self.model(features, return_logits=True)
            loss = self.criterion(logits, labels)

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

                logits = self.model(features, return_logits=True)  # Model returns raw logits
                loss = self.criterion(logits, labels) if self.criterion else None

                if loss:
                    total_loss += loss.item()
                probs = torch.sigmoid(logits)  # Convert logits to probabilities
                all_preds.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        avg_loss = total_loss / len(val_loader) if total_loss > 0 else 0.0

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        binary_preds = (all_preds > 0.5).astype(int)

        # Calculate ROC-AUC
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
            auc = 0.0

        # Calculate PR-AUC (recommended for imbalanced bioassay data)
        try:
            pr_auc = average_precision_score(all_labels, all_preds)
        except Exception as e:
            logger.warning(f"Could not calculate PR-AUC: {e}")
            pr_auc = 0.0

        # Calculate F1 and MCC (omit Accuracy, Precision, Recall as redundant)
        try:
            f1 = f1_score(all_labels, binary_preds, zero_division=0)
            mcc = matthews_corrcoef(all_labels, binary_preds)
            mcc = 0.0 if np.isnan(mcc) else mcc
        except Exception as e:
            logger.warning(f"Could not calculate classification metrics: {e}")
            f1 = mcc = 0.0

        return {
            "loss": avg_loss,
            "auc": auc,
            "pr_auc": pr_auc,
            "f1": f1,
            "mcc": mcc,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = "artifacts/models/toxicity",
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
                val_pr_auc=val_metrics.get("pr_auc", 0.0),
                val_f1=val_metrics["f1"],
                val_mcc=val_metrics["mcc"],
                learning_rate=lr,
            )

            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"AUC: {val_metrics['auc']:.4f} | "
                f"PR-AUC: {val_metrics.get('pr_auc', 0.0):.4f} | "
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
                self.best_val_auc = val_metrics["auc"]
                model_config = {
                    "input_dim": self.model.input_dim,
                    "hidden_dims": self.model.hidden_dims,
                    "dropout": self.model.dropout_rate,
                }
                self._save_checkpoint(
                    "toxicity",
                    best_model_path,
                    val_metrics,
                    model_config=model_config,
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
