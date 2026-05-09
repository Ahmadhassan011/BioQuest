"""
Configuration validation types.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class OptimizationConfig:
    """Configuration for optimization run."""

    protein_sequence: str
    seeds: List[str]
    objectives: Dict[str, float]
    max_iterations: int = 50
    batch_size: int = 50
    plateau_threshold: float = 0.001
    patience: int = 20

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.protein_sequence:
            raise ValueError("Protein sequence is required")

        if len(self.protein_sequence) < 5:
            raise ValueError("Protein sequence too short")

        if not self.seeds:
            raise ValueError("At least one seed molecule required")

        if not self.objectives:
            raise ValueError("Objectives are required")

        if sum(self.objectives.values()) <= 0:
            raise ValueError("Objective weights must sum to positive value")

        if not (1 <= self.max_iterations <= 500):
            raise ValueError("max_iterations must be between 1 and 500")

        return True


@dataclass
class ModelConfig:
    """Configuration for a model."""

    name: str
    architecture: str
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    dropout: float = 0.3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            architecture=data.get("architecture", ""),
            input_dim=data.get("input_dim", 264),
            hidden_dims=data.get("hidden_dims", [512, 256, 128]),
            output_dim=data.get("output_dim", 1),
            dropout=data.get("dropout", 0.3),
        )