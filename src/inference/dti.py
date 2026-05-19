"""
DTI (Drug-Target Interaction) inference.

Handles binding affinity prediction between molecules and proteins.
"""

import logging
from typing import List
import numpy as np
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelNotLoadedError(Exception):
    """Raised when DTI model is not loaded."""
    pass


class DTIPredictor:
    """DTI inference for binding affinity prediction."""

    def __init__(self, protein_sequence: str, models_dir: str = "artifacts/models", use_gpu: bool = False):
        """
        Initialize DTI predictor.

        Args:
            protein_sequence: Target protein sequence
            models_dir: Directory containing trained models
            use_gpu: Whether to use GPU

        Raises:
            ModelNotLoadedError: If model cannot be loaded
        """
        self.protein_sequence = protein_sequence
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self._model = None
        self._featurizer = None
        self._protein_features = None
        self._load_models(models_dir)

        if self._model is None:
            raise ModelNotLoadedError(f"DTI model not found in {models_dir}/dti/")

    def _load_models(self, models_dir: str) -> None:
        """Load DTI model and initialize featurizer."""
        from src.models.gnn_dti import GNNDTIPredictor
        from src.models.featurization import MolecularFeaturizer

        self._featurizer = MolecularFeaturizer()

        model_path = Path(models_dir) / "dti" / "best_model.pt"
        if not model_path.exists():
            logger.error(f"DTI model not found at {model_path}")
            return

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            arch_config = checkpoint.get("model_config", {})

            if arch_config:
                self._model = GNNDTIPredictor(**arch_config)
            else:
                dummy_graph = self._featurizer.featurize_molecule_graph("C")
                atom_feature_dim = dummy_graph.x.shape[1]
                self._model = GNNDTIPredictor(
                    atom_feature_dim=atom_feature_dim,
                    gcn_hidden_dim=128,
                    protein_embedding_dim=64,
                    protein_hidden_dim=128,
                    interaction_hidden_dim=256,
                    num_gcn_layers=2,
                    num_interaction_layers=3,
                    num_heads=8,
                    dropout=0.2,
                )

            self._model.to(self.device)
            state = checkpoint.get("model_state_dict", checkpoint)
            missing, unexpected = self._model.load_state_dict(state, strict=False)
            if missing or unexpected:
                logger.warning(f"DTI model state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
            self._model.eval()
            logger.info(f"DTI model loaded from {model_path}")

            prot_inds = self._featurizer.featurize_protein(self.protein_sequence)
            self._protein_features = torch.from_numpy(prot_inds).long().unsqueeze(0).to(self.device)

        except Exception as e:
            logger.error(f"Failed to load DTI model: {e}")
            self._model = None

    def predict(self, smiles: str) -> float:
        """
        Predict binding affinity for single molecule.

        Args:
            smiles: Molecule SMILES

        Returns:
            Predicted affinity

        Raises:
            ValueError: If SMILES is invalid
            RuntimeError: If prediction fails
        """
        try:
            molecule_graph = self._featurizer.featurize_molecule_graph(smiles)
            if molecule_graph is None:
                raise ValueError(f"Invalid SMILES: {smiles}")

            molecule_graph = molecule_graph.to(self.device)

            with torch.no_grad():
                affinity = self._model(molecule_graph, self._protein_features)

            return float(affinity.cpu().numpy()[0, 0])

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"DTI prediction failed: {e}") from e

    def batch_predict(self, smiles_list: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Predict binding affinities for batch of molecules.

        Args:
            smiles_list: List of SMILES strings
            batch_size: Number of molecules to process at once (default 32)

        Returns:
            Array of predicted affinities

        Raises:
            ValueError: If all SMILES are invalid
            RuntimeError: If batch prediction fails
        """
        from torch_geometric.data import Batch

        try:
            output = np.full(len(smiles_list), np.nan)

            for start_idx in range(0, len(smiles_list), batch_size):
                chunk = smiles_list[start_idx:start_idx + batch_size]
                data_list = [self._featurizer.featurize_molecule_graph(s) for s in chunk]
                valid_data = [d for d in data_list if d is not None]

                if not valid_data:
                    continue

                batched_graph = Batch.from_data_list(valid_data).to(self.device)
                num_valid = batched_graph.num_graphs
                batched_protein = self._protein_features.repeat(num_valid, 1)

                with torch.no_grad():
                    affinities = self._model(batched_graph, batched_protein)

                results = affinities.cpu().numpy().flatten()
                valid_indices_in_chunk = [i for i, d in enumerate(data_list) if d is not None]
                for local_idx, global_idx in enumerate(valid_indices_in_chunk):
                    output[start_idx + global_idx] = results[local_idx]

            if np.all(np.isnan(output)):
                raise ValueError("All SMILES strings are invalid")

            return output

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Batch DTI prediction failed: {e}") from e