"""bioquest train — model training."""

import logging
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preparation.dti import DTIDatasetPreparer, DTIGraphDataset
from src.data.preparation.toxicity import Tox21DatasetPreparer
from src.data.preparation.vae import VAEDatasetPreparer
from src.data.preparation.property import PropertyDatasetPreparer, PropertyPredictionDataset
from src.training.utils import create_data_loaders, save_training_config

logger = logging.getLogger(__name__)


def run(args) -> None:
    if args.config:
        from src.utils.pipeline import PipelineConfig
        cfg = PipelineConfig.load(args.config)
        args.epochs = cfg.epochs
        args.batch_size = cfg.batch_size
        args.learning_rate = cfg.learning_rate
        args.use_gpu = cfg.use_gpu
        args.checkpoint_dir = cfg.checkpoint_dir
        args.assay = cfg.tox_assay
        args.chembl_frac = cfg.chembl_frac
        args.dti_dataset = cfg.dti_dataset
        args.prop_dataset = cfg.prop_dataset
        args.val_split = cfg.val_split
        args.test_split = cfg.test_split
        args.use_scaffold_split = cfg.use_scaffold_split
        args.gradient_accumulation_steps = cfg.gradient_accumulation_steps
        args.kl_anneal_epochs = cfg.kl_anneal_epochs

    models = set(args.models)
    if "all" in models:
        models = {"dti", "toxicity", "vae", "property"}

    logger.info(f"Training: {', '.join(sorted(models))}")

    results: Dict = {}
    if "dti" in models:
        results["dti"] = _train_dti(args)
    if "toxicity" in models:
        results["toxicity"] = _train_toxicity(args)
    if "vae" in models:
        results["vae"] = _train_vae(args)
    if "property" in models:
        results["property"] = _train_property(args)

    summary = Path(args.checkpoint_dir) / "training_summary.json"
    save_training_config(results, str(summary))
    logger.info(f"Done. Summary: {summary}")

    if args.save_config:
        from src.utils.pipeline import PipelineConfig as PC
        PC(models=list(models), epochs=args.epochs,
           batch_size=args.batch_size, learning_rate=args.learning_rate,
           use_gpu=args.use_gpu, checkpoint_dir=args.checkpoint_dir,
           tox_assay=args.assay, chembl_frac=args.chembl_frac,
           dti_dataset=args.dti_dataset, prop_dataset=args.prop_dataset,
           val_split=args.val_split, test_split=args.test_split,
           use_scaffold_split=args.use_scaffold_split,
           gradient_accumulation_steps=args.gradient_accumulation_steps,
           kl_anneal_epochs=args.kl_anneal_epochs,
           ).save(args.save_config)


def _train_dti(args) -> Dict:
    from src.models.gnn_dti import GNNDTIPredictor
    from src.training.gnn_dti import GNNDTITrainer
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    prep = DTIDatasetPreparer()
    data_list, splits, meta = prep.prepare_dti_dataset(dataset_name=args.dti_dataset)
    train_loader, val_loader, _ = create_data_loaders(
        DTIGraphDataset(data_list), splits, batch_size=args.batch_size, dataset_type="dti")
    model = GNNDTIPredictor(atom_feature_dim=meta["atom_feature_dim"])
    trainer = GNNDTITrainer(model, device, learning_rate=args.learning_rate)
    return trainer.fit(train_loader, val_loader, epochs=args.epochs,
                       checkpoint_dir=f"{args.checkpoint_dir}/dti")


def _train_toxicity(args) -> Dict:
    from src.models.toxicity import ToxicityClassifier
    from src.training.toxicity import ToxicityClassifierTrainer
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    prep = Tox21DatasetPreparer()
    X, y, splits, meta = prep.prepare_tox21_dataset(assay=args.assay)
    train_loader, val_loader, _ = create_data_loaders(
        TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float()),
        splits, batch_size=args.batch_size, dataset_type="toxicity")
    model = ToxicityClassifier(input_dim=meta["mol_feature_dim"])
    trainer = ToxicityClassifierTrainer(model, device, learning_rate=args.learning_rate)
    return trainer.fit(train_loader, val_loader, epochs=args.epochs,
                       checkpoint_dir=f"{args.checkpoint_dir}/toxicity")


def _train_vae(args) -> Dict:
    from src.data.tokenizer import VOCAB_SIZE
    from src.models.vae import MoleculeVAE
    from src.training.vae import MoleculeVAETrainer
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    prep = VAEDatasetPreparer()
    X, splits, meta = prep.prepare_vae_dataset(sample_frac=args.chembl_frac)
    train_loader, val_loader, _ = create_data_loaders(
        TensorDataset(torch.from_numpy(X).long()),
        splits, batch_size=args.batch_size, dataset_type="vae")
    model = MoleculeVAE(vocab_size=VOCAB_SIZE, latent_dim=64)
    trainer = MoleculeVAETrainer(model, device, learning_rate=args.learning_rate)
    return trainer.fit(train_loader, val_loader, epochs=args.epochs,
                       checkpoint_dir=f"{args.checkpoint_dir}/vae")


def _train_property(args) -> Dict:
    from src.models.property import PropertyPredictor
    from src.training.property import PropertyPredictorTrainer
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    prep = PropertyDatasetPreparer()
    features, targets_dict, splits, meta = prep.prepare_property_dataset(dataset_name=args.prop_dataset)
    train_loader, val_loader, _ = create_data_loaders(
        PropertyPredictionDataset(features, targets_dict),
        splits, batch_size=args.batch_size, dataset_type="property")
    model = PropertyPredictor(input_dim=meta["mol_feature_dim"])
    trainer = PropertyPredictorTrainer(model, device, learning_rate=args.learning_rate)
    return trainer.fit(train_loader, val_loader, epochs=args.epochs,
                       checkpoint_dir=f"{args.checkpoint_dir}/property")
