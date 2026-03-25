"""Property predictor trainer module for multi-task learning."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error, r2_score

from src.models.property import PropertyPredictor
from src.training.base import Trainer

logger = logging.getLogger(__name__)


class PropertyPredictorTrainer(Trainer):
    """
    Trainer for multi-task property prediction model.

    Features:
    - Multi-task learning with separate losses for each property
    - Task-specific loss weighting
    - Uncertainty-based loss balancing
    - Comprehensive metrics tracking for all properties
    """

    def __init__(
        self,
        model: PropertyPredictor,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        task_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize property trainer.

        Args:
            model: PropertyPredictor instance
            device: Training device
            learning_rate: Initial learning rate
            weight_decay: L2 regularization coefficient
            task_weights: Task-specific loss weights
        """
        super().__init__(model, device, learning_rate, weight_decay)

        # Task weights for multi-task learning
        if task_weights is None:
            task_weights = {
                "qed": 1.0,  # Binary classification
                "sa": 1.0,  # Regression [0,1]
                "logp": 1.0,  # Regression [-1,1] (normalized)
                "mw": 1.0,  # Regression [0,1]
            }
        self.task_weights = task_weights

        # Loss functions for different tasks
        self.loss_fns = {
            "qed": nn.BCELoss(),
            "sa": nn.MSELoss(),
            "logp": nn.MSELoss(),
            "mw": nn.MSELoss(),
        }

        # Initialize training history with task-specific metrics
        history_keys = [
            "train_loss",
            "val_loss",
            "train_qed_loss",
            "train_sa_loss",
            "train_logp_loss",
            "train_mw_loss",
            "val_qed_loss",
            "val_sa_loss",
            "val_logp_loss",
            "val_mw_loss",
            "val_qed_auc",
            "val_qed_f1",
            "val_sa_mae",
            "val_sa_r2",
            "val_logp_mae",
            "val_logp_r2",
            "val_mw_mae",
            "val_mw_r2",
            "learning_rate",
        ]
        self._initialize_history(history_keys)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.task_weights.keys()}
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training Property", leave=False)
        for features, targets in pbar:
            features = features.to(self.device)

            # Move targets to device
            batch_targets = {}
            for task, target in targets.items():
                batch_targets[task] = target.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(features)

            # Calculate task-specific losses
            batch_loss = 0.0
            for task, pred in predictions.items():
                if task in batch_targets:
                    target = batch_targets[task]
                    loss = self.loss_fns[task](pred.squeeze(), target)
                    weighted_loss = self.task_weights[task] * loss
                    batch_loss += weighted_loss
                    task_losses[task] += loss.item()

            # Backward pass
            batch_loss.backward()
            self._clip_gradients(max_norm=1.0)
            self.optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{batch_loss.item():.4f}",
                    "qed": f"{task_losses['qed'] / num_batches:.4f}",
                }
            )

        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_task_losses = {
            task: loss / num_batches for task, loss in task_losses.items()
        }

        return {
            "total_loss": avg_loss,
            **avg_task_losses,
        }

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.task_weights.keys()}
        num_batches = 0

        # Dictionaries to store all predictions and targets for metric calculation
        all_preds = {task: [] for task in self.task_weights.keys()}
        all_targets = {task: [] for task in self.task_weights.keys()}

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)

                # Move targets to device
                batch_targets = {}
                for task, target in targets.items():
                    batch_targets[task] = target.to(self.device)

                # Forward pass
                predictions = self.model(features)

                # Calculate task-specific losses and collect predictions/targets
                batch_loss = 0.0
                for task, pred in predictions.items():
                    if task in batch_targets:
                        target = batch_targets[task]
                        loss = self.loss_fns[task](pred.squeeze(), target)
                        weighted_loss = self.task_weights[task] * loss
                        batch_loss += weighted_loss
                        task_losses[task] += loss.item()

                        all_preds[task].extend(pred.cpu().numpy().flatten())
                        all_targets[task].extend(target.cpu().numpy().flatten())

                total_loss += batch_loss.item()
                num_batches += 1

        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_task_losses = {
            task: loss / num_batches for task, loss in task_losses.items()
        }

        # Calculate task-specific performance metrics
        val_metrics = {"total_loss": avg_loss, **avg_task_losses}

        for task in self.task_weights.keys():
            preds = np.array(all_preds[task])
            targets = np.array(all_targets[task])

            if len(preds) == 0:
                continue  # Skip if no data for this task

            if task == "qed":  # Binary classification
                try:
                    val_metrics[f"val_{task}_auc"] = roc_auc_score(targets, preds)
                    val_metrics[f"val_{task}_f1"] = f1_score(
                        targets, (preds > 0.5).astype(int)
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not calculate {task} classification metrics: {e}"
                    )
                    val_metrics[f"val_{task}_auc"] = 0.0
                    val_metrics[f"val_{task}_f1"] = 0.0
            else:  # Regression tasks
                val_metrics[f"val_{task}_mae"] = mean_absolute_error(targets, preds)
                try:
                    val_metrics[f"val_{task}_r2"] = r2_score(targets, preds)
                except Exception as e:
                    logger.warning(
                        f"Could not calculate {task} regression metrics: {e}"
                    )
                    val_metrics[f"val_{task}_r2"] = 0.0

        return val_metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = "checkpoints/properties",
    ) -> Dict[str, Any]:
        """
        Train the property prediction model.

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

        logger.info(f"Starting property model training for {epochs} epochs")

        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)

            # Validation
            val_metrics = self.validate_epoch(val_loader)

            # Update learning rate
            lr = self._update_learning_rate(val_metrics["total_loss"])

            # Store metrics
            self._update_history(
                train_loss=train_metrics["total_loss"],
                val_loss=val_metrics["total_loss"],
                train_qed_loss=train_metrics.get("qed", 0.0),
                train_sa_loss=train_metrics.get("sa", 0.0),
                train_logp_loss=train_metrics.get("logp", 0.0),
                train_mw_loss=train_metrics.get("mw", 0.0),
                val_qed_loss=val_metrics.get("qed", 0.0),
                val_sa_loss=val_metrics.get("sa", 0.0),
                val_logp_loss=val_metrics.get("logp", 0.0),
                val_mw_loss=val_metrics.get("mw", 0.0),
                val_qed_auc=val_metrics.get("val_qed_auc", 0.0),
                val_qed_f1=val_metrics.get("val_qed_f1", 0.0),
                val_sa_mae=val_metrics.get("val_sa_mae", 0.0),
                val_sa_r2=val_metrics.get("val_sa_r2", 0.0),
                val_logp_mae=val_metrics.get("val_logp_mae", 0.0),
                val_logp_r2=val_metrics.get("val_logp_r2", 0.0),
                val_mw_mae=val_metrics.get("val_mw_mae", 0.0),
                val_mw_r2=val_metrics.get("val_mw_r2", 0.0),
                learning_rate=lr,
            )

            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train Loss: {train_metrics['total_loss']:.4f}, "
                f"Val Loss: {val_metrics['total_loss']:.4f}, "
                f"Val QED AUC: {val_metrics.get('val_qed_auc', 0.0):.4f}, "
                f"Val QED F1: {val_metrics.get('val_qed_f1', 0.0):.4f}, "
                f"Val SA MAE: {val_metrics.get('val_sa_mae', 0.0):.4f}, "
                f"Val LogP MAE: {val_metrics.get('val_logp_mae', 0.0):.4f}, "
                f"Val MW MAE: {val_metrics.get('val_mw_mae', 0.0):.4f}"
            )

            # Early stopping
            should_continue = self._check_early_stopping(
                val_metrics["total_loss"],
                is_better=lambda new, best: new < best,
                patience=early_stopping_patience,
            )

            if self.patience_counter == 0:
                self._save_checkpoint(
                    "property_predictor",
                    best_model_path,
                    val_metrics,
                )
                logger.info(
                    f"New best model saved with val_loss: {self.best_val_loss:.4f}"
                )

            if not should_continue:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Load best model
        if best_model_path.exists():
            self._load_checkpoint(best_model_path)

        results = {
            "best_val_loss": self.best_val_loss,
            "total_epochs": epoch + 1,
            "training_history": self.training_history,
            "checkpoint_path": str(best_model_path),
        }

        logger.info(
            f"Property model training complete. Best val_loss: {self.best_val_loss:.4f}"
        )
        return results
