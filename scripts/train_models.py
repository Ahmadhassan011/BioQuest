#!/usr/bin/env python3
"""
Training Script: Complete pipeline for training BioQuest custom models.

Usage:
    python scripts/train_models.py --models dti toxicity --epochs 50 --batch-size 32
    python scripts/train_models.py --all --use-gpu --checkpoint-dir ./checkpoints
"""

import argparse
import logging
import torch
import sys
from pathlib import Path
from typing import Dict
from torch.utils.data import TensorDataset

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import (
    GNNDTIPredictor,
    ToxicityClassifier,
    PropertyPredictor,
)
from src.pipelines.generation import MoleculeVAE
from src.training import (
    GNNDTITrainer,
    ToxicityClassifierTrainer,
    PropertyPredictorTrainer,
    MoleculeVAETrainer,
)
from src.data.preparers import (
    DTIDatasetPreparer,
    Tox21DatasetPreparer,
    VAEDatasetPreparer,
    DTIGraphDataset,
    PropertyDatasetPreparer,  # New import
    PropertyPredictionDataset,  # New import
)
from src.training.utils import (
    create_data_loaders,
    save_training_config,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def train_dti_model(
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    checkpoint_dir: str = "checkpoints/dti",
    use_gpu: bool = False,
    dataset: str = "DAVIS",
    gradient_accumulation_steps: int = 1,
) -> Dict:
    """
    Train DTI prediction model.
    """
    logger.info("=" * 80)
    logger.info("TRAINING DTI PREDICTION MODEL (GNN)")
    logger.info("=" * 80)

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    # Prepare data
    logger.info("Preparing DTI dataset...")
    preparer = DTIDatasetPreparer()
    data_list, splits, metadata = preparer.prepare_dti_dataset(
        dataset_name=dataset, val_split=0.1, test_split=0.1
    )

    # Create data loaders
    full_dataset = DTIGraphDataset(data_list)

    train_loader, val_loader, _ = create_data_loaders(
        full_dataset, splits=splits, batch_size=batch_size, dataset_type="dti"
    )

    # Initialize model
    model = GNNDTIPredictor(
        atom_feature_dim=metadata["atom_feature_dim"],
        protein_hidden_dim=128,
        interaction_hidden_dim=256,
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    trainer = GNNDTITrainer(model, device, learning_rate=learning_rate)
    results = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    results["metadata"] = metadata
    logger.info(f"DTI training complete. Best val_loss: {results['best_val_loss']:.4f}")

    return results


def train_toxicity_model(
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    checkpoint_dir: str = "checkpoints/toxicity",
    use_gpu: bool = False,
    assay: str = "NR-AR",
) -> Dict:
    """
    Train toxicity prediction model.
    """
    logger.info("=" * 80)
    logger.info(f"TRAINING TOXICITY PREDICTION MODEL FOR ASSAY: {assay}")
    logger.info("=" * 80)

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    logger.info(f"Using device: {device}")

    # Prepare data
    preparer = Tox21DatasetPreparer()
    X, y, splits, metadata = preparer.prepare_tox21_dataset(assay=assay)

    # Create a TensorDataset from features (X) and labels (y)
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())

    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        dataset, splits=splits, batch_size=batch_size, dataset_type="toxicity"
    )

    # Initialize model
    model = ToxicityClassifier(input_dim=metadata["mol_feature_dim"])
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = ToxicityClassifierTrainer(model, device, learning_rate=learning_rate)
    results = trainer.fit(
        train_loader, val_loader, epochs=epochs, checkpoint_dir=checkpoint_dir
    )

    results["metadata"] = metadata
    logger.info(
        f"Toxicity training complete. Best val_auc: {results['best_val_auc']:.4f}"
    )

    return results


def train_vae_model(
    batch_size: int = 64,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    checkpoint_dir: str = "checkpoints/vae",
    use_gpu: bool = False,
    kl_anneal_epochs: int = 50,
    chembl_frac: float = 0.1,
) -> Dict:
    """
    Train Variational Autoencoder model on ChEMBL data.
    """
    logger.info("=" * 80)
    logger.info("TRAINING VARIATIONAL AUTOENCODER (VAE) MODEL")
    logger.info("=" * 80)

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    logger.info(f"Using device: {device}")

    # Prepare data
    preparer = VAEDatasetPreparer()
    X, splits, metadata = preparer.prepare_vae_dataset(sample_frac=chembl_frac)

    # Create a TensorDataset from features (X)
    # VAE expects token indices as long tensors
    dataset = TensorDataset(torch.from_numpy(X).long())

    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        dataset, splits=splits, batch_size=batch_size, dataset_type="vae"
    )

    # Initialize model
    model = MoleculeVAE(
        vocab_size=len(preparer.smiles_chars),
        latent_dim=64,
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = MoleculeVAETrainer(
        model, device, learning_rate=learning_rate, kl_anneal_epochs=kl_anneal_epochs
    )
    results = trainer.fit(
        train_loader, val_loader, epochs=epochs, checkpoint_dir=checkpoint_dir
    )

    results["metadata"] = metadata
    logger.info(f"VAE training complete. Best val_loss: {results['best_val_loss']:.4f}")

    return results


def train_property_model(
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    checkpoint_dir: str = "checkpoints/properties",
    use_gpu: bool = False,
    dataset: str = "Lipophilicity_ID",  # Default dataset for property prediction
) -> Dict:
    """
    Train multi-task property prediction model.
    """
    logger.info("=" * 80)
    logger.info(f"TRAINING PROPERTY PREDICTION MODEL FOR DATASET: {dataset}")
    logger.info("=" * 80)

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    logger.info(f"Using device: {device}")

    # Prepare data
    preparer = PropertyDatasetPreparer()
    features, targets_dict, splits, metadata = preparer.prepare_property_dataset(
        dataset_name=dataset
    )

    # Create a PropertyPredictionDataset
    property_dataset = PropertyPredictionDataset(features, targets_dict)

    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        property_dataset, splits=splits, batch_size=batch_size, dataset_type="property"
    )

    # Initialize model
    model = PropertyPredictor(input_dim=metadata["mol_feature_dim"])
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = PropertyPredictorTrainer(model, device, learning_rate=learning_rate)
    results = trainer.fit(
        train_loader, val_loader, epochs=epochs, checkpoint_dir=checkpoint_dir
    )

    results["metadata"] = metadata
    logger.info(
        f"Property training complete. Best val_loss: {results['best_val_loss']:.4f}"
    )

    return results


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train custom models for AAMD")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["dti", "toxicity", "vae", "property", "all"],
        default=["all"],  # Added "property"
        help="Models to train",
    )
    parser.add_argument("--all", action="store_true", help="Train all models")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Base directory for saving checkpoints",
    )
    parser.add_argument(
        "--dti-dataset", default="DAVIS", help="PyTDC dataset for DTI training"
    )
    parser.add_argument("--tox-assay", default="NR-AR", help="Tox21 assay to train on")
    parser.add_argument(
        "--chembl-frac",
        type=float,
        default=0.1,
        help="Fraction of ChEMBL dataset to use for VAE training",
    )
    parser.add_argument(
        "--prop-dataset",
        default="lipophilicity_astrazeneca",
        help="PyTDC dataset for property training",
    )  # New argument
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of steps for gradient accumulation, to save memory.",
    )

    args = parser.parse_args()

    models_to_train = set(args.models)
    if args.all or "all" in models_to_train:
        models_to_train = {"dti", "toxicity", "vae", "property"}  # Added "property"

    logger.info(f"Training models: {', '.join(models_to_train)}")

    all_results = {}

    if "dti" in models_to_train:
        dti_results = train_dti_model(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_dir=f"{args.checkpoint_dir}/dti",
            use_gpu=args.use_gpu,
            dataset=args.dti_dataset,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        all_results["dti"] = dti_results

    if "toxicity" in models_to_train:
        tox_results = train_toxicity_model(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_dir=f"{args.checkpoint_dir}/toxicity",
            use_gpu=args.use_gpu,
            assay=args.tox_assay,
        )
        all_results["toxicity"] = tox_results

    if "vae" in models_to_train:
        vae_results = train_vae_model(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_dir=f"{args.checkpoint_dir}/vae",
            use_gpu=args.use_gpu,
            chembl_frac=args.chembl_frac,
        )
        all_results["vae"] = vae_results

    if "property" in models_to_train:  # New property training block
        prop_results = train_property_model(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_dir=f"{args.checkpoint_dir}/properties",
            use_gpu=args.use_gpu,
            dataset=args.prop_dataset,
        )
        all_results["property"] = prop_results

    # Save final summary
    summary_path = Path(args.checkpoint_dir) / "training_summary.json"
    save_training_config(all_results, str(summary_path))

    logger.info("=" * 80)
    logger.info(f"All training complete. Checkpoints saved in '{args.checkpoint_dir}'.")
    logger.info(f"Training summary saved to '{summary_path}'.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
