"""
Graph Neural Network for Drug-Target Interaction Prediction.

Predicts binding affinity between drugs and target proteins using
a GNN for molecular feature extraction and an attention-based
model for protein sequence encoding.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

from .attention import MultiHeadAttention

logger = logging.getLogger(__name__)


class GNNDTIPredictor(nn.Module):
    """
    Graph Neural Network for Drug-Target Interaction (DTI) Prediction.

    This model uses a GNN to create a graph embedding of the molecule
    and an attention-based LSTM to embed the protein sequence. These
    embeddings are then combined to predict binding affinity.
    """

    def __init__(
        self,
        atom_feature_dim: int,
        gcn_hidden_dim: int = 128,
        protein_embedding_dim: int = 64,
        protein_hidden_dim: int = 64,
        interaction_hidden_dim: int = 256,
        num_gcn_layers: int = 2,
        num_interaction_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        """
        Initialize the GNN-based DTI predictor.

        Args:
            atom_feature_dim: Dimension of atom features.
            gcn_hidden_dim: Hidden dimension for GCN layers.
            protein_embedding_dim: Dimension of the protein embedding.
            protein_hidden_dim: Hidden dimension for protein encoder.
            interaction_hidden_dim: Hidden dimension for interaction layers.
            num_gcn_layers: Number of GCN layers.
            num_interaction_layers: Number of interaction layers.
            num_heads: Number of attention heads for protein encoder.
            dropout: Dropout probability.
        """
        super().__init__()

        # Molecule Encoder (GNN)
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(atom_feature_dim, gcn_hidden_dim))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNConv(gcn_hidden_dim, gcn_hidden_dim))

        # Protein Encoder
        self.protein_embedding = nn.Embedding(
            21, protein_embedding_dim
        )  # 20 amino acids + 1 padding
        self.protein_attention = MultiHeadAttention(
            protein_embedding_dim, num_heads=num_heads, dropout=dropout
        )
        self.protein_lstm = nn.LSTM(
            protein_embedding_dim,
            protein_hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )
        self.protein_projection = nn.Linear(protein_hidden_dim * 2, protein_hidden_dim)

        # Interaction Layers
        interaction_input_dim = gcn_hidden_dim + protein_hidden_dim
        self.interaction_layers = nn.ModuleList()
        self.interaction_layers.append(
            nn.Linear(interaction_input_dim, interaction_hidden_dim)
        )
        for _ in range(num_interaction_layers - 1):
            self.interaction_layers.append(
                nn.Linear(interaction_hidden_dim, interaction_hidden_dim)
            )

        # Output Layer
        self.output_fc = nn.Linear(interaction_hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        data: Data,
        protein_indices: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the DTI predictor.

        Args:
            data: A torch_geometric.data.Data object containing 'x', 'edge_index', and 'batch'.
            protein_indices: Protein amino acid indices.
            protein_mask: Optional mask for the protein sequence.

        Returns:
            Binding affinity prediction.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        batch_size = data.num_graphs
        if protein_indices.dim() == 2:
            if protein_indices.shape[0] != batch_size:
                raise ValueError(
                    f"Expected protein_indices to have {batch_size} sequences, "
                    f"got {protein_indices.shape[0]}"
                )
        elif protein_indices.dim() == 1:
            protein_indices = protein_indices.view(batch_size, -1)
        else:
            raise ValueError(
                f"protein_indices must be 1D or 2D, got {protein_indices.dim()}D"
            )

        # Molecule encoding
        for gcn_layer in self.gcn_layers:
            x = self.relu(gcn_layer(x, edge_index))

        # Global mean pooling to get graph-level embedding
        mol_embedding = global_mean_pool(x, batch)

        # Protein encoding
        protein_hidden = self.protein_embedding(protein_indices)
        protein_attention_out, _ = self.protein_attention(
            protein_hidden, protein_hidden, protein_hidden, mask=protein_mask
        )
        protein_hidden = protein_hidden + self.dropout(protein_attention_out)

        lstm_out, _ = self.protein_lstm(protein_hidden)
        protein_representation = self.protein_projection(lstm_out[:, -1, :])

        # Combine representations
        interaction_input = torch.cat([mol_embedding, protein_representation], dim=1)

        # Interaction layers
        x = interaction_input
        for layer in self.interaction_layers:
            x = self.relu(layer(x))
            x = self.dropout(x)

        # Output
        output = self.sigmoid(self.output_fc(x))

        return output
