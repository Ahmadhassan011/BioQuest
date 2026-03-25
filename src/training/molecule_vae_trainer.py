"""VAE trainer module for molecule generation."""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.pipelines.generation import MoleculeVAE
from src.training.base import Trainer

logger = logging.getLogger(__name__)


class MoleculeVAETrainer(Trainer):
    """
    Trainer for Variational Autoencoder model.

    Features:
    - VAE training with reconstruction loss + KL divergence
    - Annealing of KL weight for better training
    - SMILES validity checking and filtering
    - Latent space visualization support
    """

    def __init__(
        self,
        model: MoleculeVAE,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        kl_weight_start: float = 0.0,
        kl_weight_end: float = 1.0,
        kl_anneal_epochs: int = 50,
    ):
        """
        Initialize VAE trainer.

        Args:
            model: MoleculeVAE instance
            device: Training device
            learning_rate: Initial learning rate
            weight_decay: L2 regularization coefficient
            kl_weight_start: Initial KL weight (beta annealing)
            kl_weight_end: Final KL weight
            kl_anneal_epochs: Number of epochs for KL annealing
        """
        super().__init__(model, device, learning_rate, weight_decay)

        # KL annealing parameters (beta-annealing)
        self.kl_weight_start = kl_weight_start
        self.kl_weight_end = kl_weight_end
        self.kl_anneal_epochs = kl_anneal_epochs

        # Initialize training history with VAE-specific metrics
        self._initialize_history(
            [
                "train_loss",
                "val_loss",
                "train_recon_loss",
                "val_recon_loss",
                "train_kl_loss",
                "val_kl_loss",
                "kl_weight",
                "val_accuracy",
                "learning_rate",
            ]
        )

    def get_kl_weight(self, epoch: int) -> float:
        """
        Calculate KL weight for current epoch with beta annealing.

        Args:
            epoch: Current epoch number

        Returns:
            KL weight for this epoch
        """
        if epoch >= self.kl_anneal_epochs:
            return self.kl_weight_end

        # Linear annealing from start to end
        progress = epoch / self.kl_anneal_epochs
        return self.kl_weight_start + progress * (
            self.kl_weight_end - self.kl_weight_start
        )

    def vae_loss(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate VAE loss: reconstruction + KL divergence.

        Args:
            reconstructed: Reconstructed SMILES logits
            original: Original SMILES indices
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
            kl_weight: Weight for KL divergence term (beta)

        Returns:
            Tuple of (total_loss, reconstruction_loss, kl_loss)
        """
        # Reconstruction loss (cross-entropy for discrete SMILES)
        recon_loss = nn.CrossEntropyLoss()(
            reconstructed.view(-1, reconstructed.size(-1)), original.view(-1)
        )

        # KL divergence loss (analytical form for Gaussian)
        # KL(N(mu, sigma), N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss with beta weighting
        total_loss = recon_loss + kl_weight * kl_loss

        return total_loss, recon_loss, kl_loss

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        # Get current KL weight (beta annealing)
        kl_weight = self.get_kl_weight(epoch)

        pbar = tqdm(train_loader, desc="Training VAE", leave=False)
        for smiles_batch in pbar:
            if isinstance(smiles_batch, (list, tuple)):
                smiles_batch = smiles_batch[0]

            smiles_batch = smiles_batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            reconstructed, mu, logvar = self.model(smiles_batch)

            # Calculate losses
            batch_loss, recon_loss, kl_loss = self.vae_loss(
                reconstructed, smiles_batch, mu, logvar, kl_weight
            )

            # Backward pass
            batch_loss.backward()
            self._clip_gradients(max_norm=1.0)
            self.optimizer.step()

            total_loss += batch_loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{batch_loss.item():.4f}",
                    "beta": f"{kl_weight:.4f}",
                }
            )

        # Calculate averages
        return {
            "total_loss": total_loss / num_batches,
            "recon_loss": total_recon_loss / num_batches,
            "kl_loss": total_kl_loss / num_batches,
            "kl_weight": kl_weight,
        }

    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        correct_reconstructions = 0
        total_reconstructions = 0

        # Get current KL weight (beta annealing)
        kl_weight = self.get_kl_weight(epoch)

        with torch.no_grad():
            for smiles_batch in val_loader:
                if isinstance(smiles_batch, (list, tuple)):
                    smiles_batch = smiles_batch[0]

                smiles_batch = smiles_batch.to(self.device)

                # Forward pass
                reconstructed, mu, logvar = self.model(smiles_batch)

                # Calculate losses
                batch_loss, recon_loss, kl_loss = self.vae_loss(
                    reconstructed, smiles_batch, mu, logvar, kl_weight
                )

                total_loss += batch_loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                num_batches += 1

                # Calculate reconstruction accuracy (token-level accuracy)
                pred_tokens = torch.argmax(reconstructed, dim=-1)
                correct = (pred_tokens == smiles_batch).sum().item()
                correct_reconstructions += correct
                total_reconstructions += smiles_batch.numel()

        # Calculate averages
        accuracy = (
            correct_reconstructions / total_reconstructions
            if total_reconstructions > 0
            else 0.0
        )

        return {
            "total_loss": total_loss / num_batches,
            "recon_loss": total_recon_loss / num_batches,
            "kl_loss": total_kl_loss / num_batches,
            "kl_weight": kl_weight,
            "accuracy": accuracy,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        checkpoint_dir: str = "checkpoints/vae",
    ) -> Dict[str, Any]:
        """
        Train the VAE model.

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

        logger.info(
            f"Starting VAE training for {epochs} epochs "
            f"with KL annealing ({self.kl_anneal_epochs} epochs)"
        )

        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validation
            val_metrics = self.validate_epoch(val_loader, epoch)

            # Update learning rate
            lr = self._update_learning_rate(val_metrics["total_loss"])

            # Store metrics
            self._update_history(
                train_loss=train_metrics["total_loss"],
                val_loss=val_metrics["total_loss"],
                train_recon_loss=train_metrics["recon_loss"],
                val_recon_loss=val_metrics["recon_loss"],
                train_kl_loss=train_metrics["kl_loss"],
                val_kl_loss=val_metrics["kl_loss"],
                kl_weight=train_metrics["kl_weight"],
                val_accuracy=val_metrics.get("accuracy", 0.0),
                learning_rate=lr,
            )

            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train Loss: {train_metrics['total_loss']:.4f}, "
                f"Val Loss: {val_metrics['total_loss']:.4f}, "
                f"KL Weight: {train_metrics['kl_weight']:.4f}, "
                f"Accuracy: {val_metrics.get('accuracy', 0.0):.4f}"
            )

            # Early stopping
            should_continue = self._check_early_stopping(
                val_metrics["total_loss"],
                is_better=lambda new, best: new < best,
                patience=early_stopping_patience,
            )

            if self.patience_counter == 0:
                self._save_checkpoint(
                    "molecule_vae",
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

        logger.info(f"VAE training complete. Best val_loss: {self.best_val_loss:.4f}")
        return results
