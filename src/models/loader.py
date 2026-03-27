"""
Model Utils Module: Utilities for model management, inference, and versioning.

This module provides:
- Model loading and inference wrappers
- Model versioning and registry management
- Device management and optimization
- Model performance evaluation utilities
- Inference caching for efficiency
"""

import logging
import json
import torch
from typing import Dict, Optional, List, Any
from pathlib import Path
import numpy as np
from datetime import datetime

from .gnn_dti import GNNDTIPredictor
from .toxicity import ToxicityClassifier
from .property import PropertyPredictor
from .featurization import MolecularFeaturizer
from torch_geometric.data import Batch

logger = logging.getLogger(__name__)


# ============================================================================
# MODEL LOADER
# ============================================================================


class ModelLoader:
    """
    Loads and manages custom trained models with version control.

    Features:
    - Automatic device detection and management
    - Model checkpoint loading with error handling
    - Version tracking and metadata
    - Inference caching
    """

    def __init__(self, models_dir: str = "trained_models", use_gpu: bool = False):
        """
        Initialize model loader.

        Args:
            models_dir: Directory containing trained models
            use_gpu: Whether to use GPU if available
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        )
        self.loaded_models = {}
        self.featurizer = MolecularFeaturizer()

        logger.info(f"ModelLoader initialized on device: {self.device}")

    def load_dti_model(
        self,
        model_path: Optional[str] = None,
        model_name: str = "gnn_dti_best",
    ) -> GNNDTIPredictor:
        """
        Load DTI prediction model.

        Args:
            model_path: Path to model checkpoint
            model_name: Name of model variant

        Returns:
            Loaded GNNDTIPredictor model
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        if model_path is None:
            model_path = self.models_dir / f"{model_name}.pt"

        model_path = Path(model_path)

        # Get atom feature dimension from a dummy molecule
        dummy_graph = self.featurizer.featurize_molecule_graph("C")
        atom_feature_dim = dummy_graph.x.shape[1]

        if not model_path.exists():
            logger.error(f"DTI model checkpoint not found at {model_path}")
            raise FileNotFoundError(f"DTI model checkpoint not found at {model_path}")
        else:
            # Initialize model with same architecture used during training
            model = GNNDTIPredictor(
                atom_feature_dim=atom_feature_dim,
                gcn_hidden_dim=128,
                protein_embedding_dim=64,
                protein_hidden_dim=128,  # MUST MATCH training.py which uses 128
                interaction_hidden_dim=256,
                num_gcn_layers=2,
                num_interaction_layers=3,
                num_heads=8,
                dropout=0.2,
            )
            model.to(self.device)

            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            state = checkpoint.get("model_state_dict", checkpoint)
            try:
                model.load_state_dict(state)
                logger.info(f"Loaded DTI model from {model_path} (strict load)")
            except Exception as e:
                logger.warning(
                    f"Strict state_dict load failed for DTI model: {e}. Trying non-strict load."
                )
                load_res = model.load_state_dict(state, strict=False)
                logger.info(
                    f"DTI model loaded with non-strict load. Missing keys: {load_res.missing_keys}, "
                    f"Unexpected keys: {load_res.unexpected_keys}"
                )

        model.eval()
        self.loaded_models[model_name] = model
        return model

    def load_toxicity_model(
        self,
        model_path: Optional[str] = None,
        model_name: str = "toxicity_best",
    ) -> ToxicityClassifier:
        """Load toxicity prediction model."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        if model_path is None:
            model_path = self.models_dir / f"{model_name}.pt"

        model_path = Path(model_path)

        if not model_path.exists():
            logger.error(f"Toxicity model checkpoint not found at {model_path}")
            raise FileNotFoundError(
                f"Toxicity model checkpoint not found at {model_path}"
            )
        else:
            model = ToxicityClassifier(
                input_dim=264,
                hidden_dims=[512, 256, 128, 64],
                dropout=0.3,
            )
            model.to(self.device)

            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            state = checkpoint.get("model_state_dict", checkpoint)
            try:
                model.load_state_dict(state)
                logger.info(f"Loaded toxicity model from {model_path} (strict load)")
            except Exception as e:
                logger.warning(
                    f"Strict state_dict load failed for toxicity model: {e}. Trying non-strict load."
                )
                load_res = model.load_state_dict(state, strict=False)
                logger.info(
                    f"Toxicity model loaded with non-strict load. Missing keys: {load_res.missing_keys}, "
                    f"Unexpected keys: {load_res.unexpected_keys}"
                )

        model.eval()
        self.loaded_models[model_name] = model
        return model

    def load_property_model(
        self,
        model_path: Optional[str] = None,
        model_name: str = "properties_best",
    ) -> PropertyPredictor:
        """Load multi-task property prediction model."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        if model_path is None:
            model_path = self.models_dir / f"{model_name}.pt"

        model_path = Path(model_path)

        if not model_path.exists():
            logger.error(f"Property model checkpoint not found at {model_path}")
            raise FileNotFoundError(
                f"Property model checkpoint not found at {model_path}"
            )
        else:
            model = PropertyPredictor(
                input_dim=264,
                shared_hidden_dim=256,
                task_hidden_dim=128,
            )
            model.to(self.device)

            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            state = checkpoint.get("model_state_dict", checkpoint)
            try:
                model.load_state_dict(state)
                logger.info(f"Loaded property model from {model_path} (strict load)")
            except Exception as e:
                logger.warning(
                    f"Strict state_dict load failed for property model: {e}. Trying non-strict load."
                )
                load_res = model.load_state_dict(state, strict=False)
                logger.info(
                    f"Property model loaded with non-strict load. Missing keys: {load_res.missing_keys}, "
                    f"Unexpected keys: {load_res.unexpected_keys}"
                )

        model.eval()
        self.loaded_models[model_name] = model
        return model


# ============================================================================
# CUSTOM PREDICTOR WRAPPER
# ============================================================================


class CustomModelPredictor:
    """
    Wrapper for making predictions using custom trained models.

    This replaces the heuristic predictors with neural network models,
    providing better performance and learned representations.
    """

    def __init__(
        self,
        protein_sequence: str,
        models_dir: str = "trained_models",
        use_gpu: bool = False,
    ):
        """
        Initialize custom model predictor.

        Args:
            protein_sequence: Target protein sequence
            models_dir: Directory with trained models
            use_gpu: Whether to use GPU
        """
        self.protein_sequence = protein_sequence
        self.device = torch.device(
            "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        )

        self.loader = ModelLoader(models_dir=models_dir, use_gpu=use_gpu)
        self.featurizer = MolecularFeaturizer()

        # Load models from subdirectories
        dti_path = Path(models_dir) / "dti" / "dti.pt"
        toxicity_path = Path(models_dir) / "toxicity" / "toxicity.pt"
        property_path = Path(models_dir) / "properties" / "properties.pt"

        self.dti_model = self.loader.load_dti_model(model_path=str(dti_path))
        self.toxicity_model = self.loader.load_toxicity_model(
            model_path=str(toxicity_path)
        )
        self.property_model = self.loader.load_property_model(
            model_path=str(property_path)
        )

        # Cache protein features as integer indices for embedding (LongTensor)
        prot_inds = self.featurizer.featurize_protein(protein_sequence)
        self.protein_features = (
            torch.from_numpy(prot_inds).long().unsqueeze(0).to(self.device)
        )

        logger.info("CustomModelPredictor initialized")

    def predict_dti_affinity(self, smiles: str) -> float:
        """
        Predict binding affinity using neural network.

        Args:
            smiles: Molecule SMILES

        Returns:
            Predicted affinity (0-1)
        """
        try:
            molecule_graph = self.featurizer.featurize_molecule_graph(smiles)
            if molecule_graph is None:
                logger.warning(f"Could not featurize SMILES: {smiles}, returning 0.5")
                return 0.5

            molecule_graph = molecule_graph.to(self.device)

            with torch.no_grad():
                affinity = self.dti_model(molecule_graph, self.protein_features)

            return float(affinity.cpu().numpy()[0, 0])

        except Exception as e:
            logger.warning(f"DTI prediction failed: {e}, returning fallback 0.5")
            return 0.5

    def predict_toxicity(self, smiles: str) -> float:
        """
        Predict toxicity using neural network.

        Args:
            smiles: Molecule SMILES

        Returns:
            Predicted toxicity (0-1, where 1 is most toxic)
        """
        try:
            mol_features = (
                torch.from_numpy(self.featurizer.featurize_molecule(smiles))
                .float()
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                toxicity = self.toxicity_model(mol_features)

            return float(toxicity.cpu().numpy()[0, 0])

        except Exception as e:
            logger.warning(f"Toxicity prediction failed: {e}, returning fallback 0.5")
            return 0.5

    def predict_properties(self, smiles: str) -> Dict[str, float]:
        """
        Predict multiple properties using multi-task network.

        Args:
            smiles: Molecule SMILES

        Returns:
            Dictionary of property predictions
        """
        try:
            mol_features = (
                torch.from_numpy(self.featurizer.featurize_molecule(smiles))
                .float()
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                predictions = self.property_model(mol_features)

            return {
                "qed": float(predictions["qed"].cpu().numpy()[0, 0]),
                "sa": float(predictions["sa"].cpu().numpy()[0, 0]),
                "logp": float(predictions["logp"].cpu().numpy()[0, 0]),
                "mw": float(predictions["mw"].cpu().numpy()[0, 0]),
            }

        except Exception as e:
            logger.warning(f"Property prediction failed: {e}, returning fallback")
            return {"qed": 0.5, "sa": 0.5, "logp": 0.0, "mw": 0.5}

    def batch_predict_affinity(self, smiles_list: List[str]) -> np.ndarray:
        """Predict affinities for multiple molecules in a batch."""
        try:
            data_list = [
                self.featurizer.featurize_molecule_graph(s) for s in smiles_list
            ]
            valid_data = [d for d in data_list if d is not None]

            if not valid_data:
                return np.full(len(smiles_list), 0.5)

            batched_graph = Batch.from_data_list(valid_data).to(self.device)

            batch_size = batched_graph.num_graphs
            batched_protein = self.protein_features.repeat(batch_size, 1)

            with torch.no_grad():
                affinities = self.dti_model(batched_graph, batched_protein)

            # Handle invalid SMILES by inserting fallback values
            results = np.full(len(smiles_list), 0.5)
            valid_indices = [i for i, d in enumerate(data_list) if d is not None]
            results[valid_indices] = affinities.cpu().numpy().flatten()

            return results

        except Exception as e:
            logger.warning(f"Batch DTI prediction failed: {e}, returning fallback")
            return np.full(len(smiles_list), 0.5)

    def batch_predict_toxicity(self, smiles_list: List[str]) -> np.ndarray:
        """Predict toxicities for multiple molecules in a batch."""
        try:
            mol_features_batch = (
                torch.from_numpy(self.featurizer.batch_featurize_molecules(smiles_list))
                .float()
                .to(self.device)
            )

            with torch.no_grad():
                toxicities = self.toxicity_model(mol_features_batch)

            return toxicities.cpu().numpy().flatten()

        except Exception as e:
            logger.warning(f"Batch toxicity prediction failed: {e}, returning fallback")
            return np.full(len(smiles_list), 0.5)

    def batch_predict_properties(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Predict multiple properties for a batch of molecules using multi-task network.

        Args:
            smiles_list: List of Molecule SMILES

        Returns:
            Dictionary of property predictions (each as a NumPy array)
        """
        try:
            mol_features_batch = (
                torch.from_numpy(self.featurizer.batch_featurize_molecules(smiles_list))
                .float()
                .to(self.device)
            )

            with torch.no_grad():
                predictions = self.property_model(mol_features_batch)

            return {
                "qed": predictions["qed"].cpu().numpy().flatten(),
                "sa": predictions["sa"].cpu().numpy().flatten(),
                "logp": predictions["logp"].cpu().numpy().flatten(),
                "mw": predictions["mw"].cpu().numpy().flatten(),
            }

        except Exception as e:
            logger.warning(f"Batch property prediction failed: {e}, returning fallback")
            # Return fallback arrays of appropriate size
            num_molecules = len(smiles_list)
            return {
                "qed": np.full(num_molecules, 0.5),
                "sa": np.full(num_molecules, 0.5),
                "logp": np.full(num_molecules, 0.0),
                "mw": np.full(num_molecules, 0.5),
            }

    def predict_all_properties(self, smiles: str) -> Dict[str, float]:
        """
        Predict all properties (affinity, toxicity, QED, SA).

        Args:
            smiles: Molecule SMILES

        Returns:
            Dictionary with all property predictions
        """
        affinity = self.predict_dti_affinity(smiles)
        toxicity = self.predict_toxicity(smiles)
        properties = self.predict_properties(smiles)

        return {
            "affinity": affinity,
            "toxicity": toxicity,
            "qed": properties.get("qed", 0.5),
            "sa": properties.get("sa", 0.5),
        }


# ============================================================================
# MODEL EVALUATION
# ============================================================================


class ModelEvaluator:
    """
    Evaluates model performance on test sets.

    Metrics:
    - Regression (DTI): RMSE, MAE, R²
    - Classification (Toxicity): AUC, Accuracy, F1
    """

    @staticmethod
    def evaluate_dti_model(
        model: GNNDTIPredictor,
        test_loader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Evaluate DTI model on test set.

        Args:
            model: Trained DTI model
            test_loader: Test data loader
            device: Device

        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for mol_graphs, prot_features, affinities in test_loader:
                mol_graphs = {k: v.to(device) for k, v in mol_graphs.items()}
                prot_features = prot_features.to(device)

                predictions = model(mol_graphs, prot_features)
                all_preds.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(affinities.numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        mae = np.mean(np.abs(all_preds - all_targets))
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "num_samples": len(all_preds),
        }

    @staticmethod
    def evaluate_toxicity_model(
        model: ToxicityClassifier,
        test_loader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluate toxicity model on test set."""
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                predictions = model(features)
                all_preds.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        auc = roc_auc_score(all_labels, all_preds)
        accuracy = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
        f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))

        return {
            "auc": float(auc),
            "accuracy": float(accuracy),
            "f1": float(f1),
            "num_samples": len(all_preds),
        }
