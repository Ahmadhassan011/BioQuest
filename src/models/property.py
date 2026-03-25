"""
Multi-Task Property Predictor Module.

Neural network for predicting multiple molecular properties simultaneously
using shared representations and task-specific decoders.
"""

import logging
from typing import Dict
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PropertyPredictor(nn.Module):
    """
    Multi-task neural network for predicting multiple molecular properties.

    Architecture:
    - Shared encoder: Common feature representation learning
    - Property-specific decoders: Task-specific prediction heads
    - Multi-task learning: Simultaneous optimization of all objectives

    Properties:
    - QED (Drug-likeness): Binary classifier
    - SA (Synthetic Accessibility): Regression (0-1)
    - LogP (Lipophilicity): Regression (-3 to +5 normalized)
    - MW (Molecular Weight): Regression (0-1 normalized)
    """

    def __init__(
        self,
        input_dim: int = 264,
        shared_hidden_dim: int = 256,
        task_hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        """
        Initialize property predictor.

        Args:
            input_dim: Input feature dimension
            shared_hidden_dim: Shared encoder hidden dimension
            task_hidden_dim: Task-specific decoder hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        # Shared encoder with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, shared_hidden_dim),
            nn.BatchNorm1d(shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden_dim, shared_hidden_dim),
            nn.BatchNorm1d(shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # QED decoder (drug-likeness)
        self.qed_decoder = nn.Sequential(
            nn.Linear(shared_hidden_dim, task_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(task_hidden_dim, 1),
            nn.Sigmoid(),
        )

        # SA decoder (synthetic accessibility)
        self.sa_decoder = nn.Sequential(
            nn.Linear(shared_hidden_dim, task_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(task_hidden_dim, 1),
            nn.Sigmoid(),
        )

        # LogP decoder
        self.logp_decoder = nn.Sequential(
            nn.Linear(shared_hidden_dim, task_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(task_hidden_dim, 1),
            nn.Tanh(),  # Range -1 to 1, represents -3 to +5 logp
        )

        # MW decoder
        self.mw_decoder = nn.Sequential(
            nn.Linear(shared_hidden_dim, task_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(task_hidden_dim, 1),
            nn.Sigmoid(),  # Normalized MW (0-1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of property predictor.

        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            Dictionary of property predictions
        """
        # Shared encoding
        shared_representation = self.encoder(x)

        # Task-specific predictions
        predictions = {
            "qed": self.qed_decoder(shared_representation),
            "sa": self.sa_decoder(shared_representation),
            "logp": self.logp_decoder(shared_representation),
            "mw": self.mw_decoder(shared_representation),
        }

        return predictions
