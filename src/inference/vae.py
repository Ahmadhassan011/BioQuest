"""
VAE inference.

Handles molecule generation using Variational Autoencoder.
"""

import logging
from typing import List, Optional
import torch
from pathlib import Path

from src.data.tokenizer import indices_to_smiles, smiles_to_indices, VOCAB_SIZE

logger = logging.getLogger(__name__)


class VAEGenerator:
    """VAE inference for molecule generation from latent space."""

    def __init__(self, models_dir: str = "artifacts/models", use_gpu: bool = False):
        """
        Initialize VAE generator.

        Args:
            models_dir: Directory containing trained models
            use_gpu: Whether to use GPU
        """
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self._model = None
        self._load_models(models_dir)

    def _load_models(self, models_dir: str) -> None:
        """Load VAE model."""
        try:
            from src.models.vae import MoleculeVAE

            model_path = Path(models_dir) / "vae" / "best_model.pt"

            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                arch_config = checkpoint.get("model_config", {})

                default_vocab = arch_config.get("vocab_size", None) or VOCAB_SIZE
                self._model = MoleculeVAE(
                    vocab_size=default_vocab,
                    embedding_dim=arch_config.get("embedding_dim", 128),
                    hidden_dim=arch_config.get("hidden_dim", 256),
                    latent_dim=arch_config.get("latent_dim", 64),
                )
                self._model.to(self.device)
                state = checkpoint.get("model_state_dict", checkpoint)
                self._model.load_state_dict(state, strict=True)
                self._model.eval()
                logger.info(f"VAE model loaded from {model_path}")
            else:
                logger.warning(f"VAE model not found at {model_path}")

        except Exception as e:
            logger.warning(f"Could not load VAE model: {e}")
            self._model = None

    def generate_from_latent_space(self, num_molecules: int = 10) -> List[str]:
        """Alias for generate()."""
        return self.generate(num_molecules)

    def generate_from_seeds(self, seeds: List[str], num_molecules: int = 10) -> List[str]:
        """Generate molecules similar to seeds via latent-space perturbation."""
        from rdkit import Chem
        generated = []
        for i in range(num_molecules):
            seed = seeds[i % len(seeds)]
            z = self.encode(seed)
            if z is not None:
                z_noisy = z + torch.randn_like(z) * 0.5
                new_smiles = self.decode(z_noisy)
                if new_smiles:
                    mol = Chem.MolFromSmiles(new_smiles)
                    if mol:
                        generated.append(Chem.MolToSmiles(mol))
                        continue
            generated.append(seed)
        return generated[:num_molecules]

    def generate_hybrid(self, seeds: List[str], num_molecules: int = 10) -> List[str]:
        """Mix of VAE and seed-based generation."""
        half = num_molecules // 2
        from_seeds = self.generate_from_seeds(seeds, half)
        from_vae = self.generate(num_molecules - len(from_seeds))
        return from_seeds + from_vae

    @staticmethod
    def get_unique_molecules(molecules: List[str]) -> List[str]:
        """Canonicalize and deduplicate SMILES."""
        from rdkit import Chem
        unique = set()
        for smi in molecules:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                unique.add(Chem.MolToSmiles(mol))
        return list(unique)

    def encode(self, smiles: str) -> Optional[torch.Tensor]:
        """
        Encode SMILES to latent space.

        Args:
            smiles: Molecule SMILES

        Returns:
            Latent representation or None
        """
        if self._model is None:
            return None

        try:
            token_indices = self._smiles_to_indices(smiles)
            if token_indices is None:
                return None

            tensor = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0).to(self.device)

            with torch.no_grad():
                mu, logvar = self._model.encode(tensor)
                z = self._model.reparameterize(mu, logvar)

            return z

        except Exception as e:
            logger.warning(f"VAE encoding failed: {e}")
            return None

    def decode(self, z: torch.Tensor, max_length: int = 100) -> Optional[str]:
        """
        Decode latent representation to SMILES.

        Args:
            z: Latent tensor
            max_length: Maximum SMILES length

        Returns:
            Generated SMILES or None
        """
        if self._model is None:
            return None

        try:
            with torch.no_grad():
                token_ids = self._model._decode_tokens(z, max_length)

            smiles = self._indices_to_smiles(token_ids)
            return smiles

        except Exception as e:
            logger.warning(f"VAE decoding failed: {e}")
            return None

    def generate(self, num_molecules: int = 10) -> List[str]:
        """
        Generate molecules from random latent samples.

        Args:
            num_molecules: Number of molecules to generate

        Returns:
            List of generated SMILES
        """
        if self._model is None:
            return []

        try:
            from rdkit import Chem

            z = torch.randn(num_molecules, self._model.latent_dim).to(self.device)
            generated = []

            for i in range(num_molecules):
                smiles = self.decode(z[i:i+1])
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        canonical = Chem.MolToSmiles(mol)
                        generated.append(canonical)

            return generated

        except Exception as e:
            logger.warning(f"VAE generation failed: {e}")
            return []

    def _smiles_to_indices(self, smiles: str) -> Optional[List[int]]:
        """Convert SMILES to token indices using shared tokenizer."""
        arr = smiles_to_indices(smiles, max_len=100)
        return arr.tolist()

    def _indices_to_smiles(self, indices: torch.Tensor) -> str:
        """Convert token indices to SMILES using shared tokenizer."""
        idx_arr = indices.cpu().numpy().flatten()
        return indices_to_smiles(idx_arr)