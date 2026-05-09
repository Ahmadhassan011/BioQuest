"""
VAE Module: Variational Autoencoder for molecule generation.

This module provides:
- Variational Autoencoder (VAE) for molecule generation in latent space
- Character-level encoder/decoder using GRU
- Reparameterization trick for sampling

Architecture:
- Encoder: Embedding → GRU → Linear(mu) + Linear(logvar)
- Decoder: Latent → Linear → GRU → Linear(vocab)
"""

import logging
from typing import Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MoleculeVAE(nn.Module):
    """Variational Autoencoder for molecule generation."""

    def __init__(
        self,
        vocab_size: int = 100,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 64,
    ):
        """
        Initialize Molecule VAE.

        Args:
            vocab_size: Size of SMILES character vocabulary
            embedding_dim: Embedding dimension for characters
            hidden_dim: Hidden dimension for encoder/decoder
            latent_dim: Latent space dimension
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.latent_to_emb = nn.Linear(latent_dim, embedding_dim)
        self.decoder_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode molecule to latent space.

        Args:
            x: Input tensor of SMILES encoded as integers

        Returns:
            Tuple of (mu, logvar) representing mean and log-variance
        """
        embedded = self.embedding(x)
        _, hidden = self.encoder_gru(embedded)
        mu = self.fc_mu(hidden.squeeze(0))
        logvar = self.fc_logvar(hidden.squeeze(0))
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.

        Args:
            mu: Mean of distribution
            logvar: Log-variance of distribution

        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode_tokens(self, z: torch.Tensor, max_len: int = 100) -> torch.Tensor:
        """
        Decode latent vector to molecule tokens (for inference).

        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            max_len: Maximum length of generated molecule

        Returns:
            Generated molecule as encoded tensor of token IDs, shape (batch_size, max_len)
        """
        logits = self.decode(z, max_len)
        return logits.argmax(dim=-1)

    def decode(self, z: torch.Tensor, max_len: int = 100) -> torch.Tensor:
        """
        Decode latent vector to sequence of logits (autoregressive inference).

        Feeds the previously predicted token (via argmax + embedding) as input
        to each subsequent step, enabling proper autoregressive generation.

        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            max_len: Maximum length of the decoded sequence

        Returns:
            Logits for each token at each position, shape (batch_size, max_len, vocab_size)
        """
        hidden = torch.tanh(self.decoder_input(z)).unsqueeze(0)

        all_logits = []
        current_input = self.latent_to_emb(z).unsqueeze(1)

        for _ in range(max_len):
            gru_out, hidden = self.decoder_gru(current_input, hidden)
            logits = self.fc_out(gru_out.squeeze(1))
            all_logits.append(logits.unsqueeze(1))
            pred_tokens = logits.argmax(dim=-1)
            current_input = self.embedding(pred_tokens).unsqueeze(1)

        return torch.cat(all_logits, dim=1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE with teacher forcing.

        At each decoding step the ground-truth token (from x) is fed as
        the next input instead of the model's own prediction.  This provides
        stronger gradient signal during training.

        Args:
            x: Input SMILES tensor (used as target for reconstruction during training)

        Returns:
            Tuple of (reconstructed_logits, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        batch_size, max_len = x.size()
        hidden = torch.tanh(self.decoder_input(z)).unsqueeze(0)

        all_logits = []
        current_input = self.latent_to_emb(z).unsqueeze(1)

        for t in range(max_len):
            gru_out, hidden = self.decoder_gru(current_input, hidden)
            logits = self.fc_out(gru_out.squeeze(1))
            all_logits.append(logits.unsqueeze(1))
            current_input = self.embedding(x[:, t]).unsqueeze(1)

        reconstructed_logits = torch.cat(all_logits, dim=1)
        return reconstructed_logits, mu, logvar