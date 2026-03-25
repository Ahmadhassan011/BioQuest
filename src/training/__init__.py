"""
Training module for model training and evaluation.

This module provides specialized trainers for each model type:
- GNNDTITrainer: Drug-Target Interaction prediction
- ToxicityClassifierTrainer: Toxicity classification
- PropertyPredictorTrainer: Multi-task property prediction
- MoleculeVAETrainer: Molecule generation with VAE

All trainers inherit from Trainer base class which provides:
- Optimizer and learning rate scheduling
- Early stopping with patience
- Gradient clipping
- Checkpoint management
- Training history tracking

Example Usage:
    >>> from src.training import GNNDTITrainer
    >>> trainer = GNNDTITrainer(model, device, learning_rate=1e-3)
    >>> history = trainer.fit(train_loader, val_loader, epochs=50)
    >>> trainer._save_checkpoint('gnn_dti', 'trained_models/', metadata={'best_loss': history['val_loss'][-1]})
"""

# Base trainer class
from src.training.base import Trainer

# Specialized trainers
from src.training.gnn_dti_trainer import GNNDTITrainer
from src.training.toxicity_classifier_trainer import ToxicityClassifierTrainer
from src.training.property_predictor_trainer import PropertyPredictorTrainer
from src.training.molecule_vae_trainer import MoleculeVAETrainer

# Training utilities
from src.training.utils import (
    create_data_loaders,
    save_training_config,
)

__all__ = [
    # Base trainer
    "Trainer",
    # Specialized trainers
    "GNNDTITrainer",
    "ToxicityClassifierTrainer",
    "PropertyPredictorTrainer",
    "MoleculeVAETrainer",
    # Utilities
    "create_data_loaders",
    "save_training_config",
]
