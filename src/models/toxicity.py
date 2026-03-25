"""
Toxicity Classifier Module.

Deep neural network for predicting molecular toxicity using
feature-level attention and residual connections.
"""

import logging
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ToxicityClassifier(nn.Module):
    """
    Deep Neural Network for Toxicity Prediction.

    Architecture:
    - Feature extraction: Dense layers with batch normalization
    - Deep representation: 4+ hidden layers with residual connections
    - Attention over features: Self-attention for feature importance weighting
    - Classification: Sigmoid output for binary toxicity prediction

    WOW Factor:
    - Feature-level attention mechanism
    - Deep residual architecture for complex patterns
    - Batch normalization for training stability
    - Ensemble-ready architecture
    """

    def __init__(
        self,
        input_dim: int = 264,  # Morgan fingerprints + descriptors
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        """
        Initialize toxicity classifier.

        Args:
            input_dim: Input feature dimension
            hidden_dims: Dimensions of hidden layers
            dropout: Dropout probability
            use_attention: Whether to use feature attention
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]

        self.input_dim = input_dim
        self.dropout_rate = dropout
        self.use_attention = use_attention

        # Feature attention layer
        if use_attention:
            self.feature_attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, input_dim),
                nn.Sigmoid(),
            )

        # Deep residual layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.projections = nn.ModuleDict()

        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout))

            if prev_dim != hidden_dim:
                self.projections[str(i)] = nn.Linear(prev_dim, hidden_dim)

            prev_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of toxicity classifier.

        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            Toxicity predictions of shape (batch_size, 1)
        """
        # Feature attention
        if self.use_attention:
            attention_weights = self.feature_attention(x)
            x = x * attention_weights

        # Deep residual processing
        hidden = x
        for i, (layer, bn, dropout) in enumerate(
            zip(self.layers, self.batch_norms, self.dropout_layers)
        ):
            residual = hidden

            hidden = layer(hidden)
            hidden = bn(hidden)
            hidden = F.relu(hidden)
            hidden = dropout(hidden)

            # Add residual connection
            if str(i) in self.projections:
                residual = self.projections[str(i)](residual)

            if residual.shape == hidden.shape:
                hidden = hidden + residual

        # Output with sigmoid
        output = torch.sigmoid(self.output_layer(hidden))

        return output
