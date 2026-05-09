"""
Data preparation base classes and utilities.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Any, Dict, List, Union
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class DatasetResult:
    """Standardized container for dataset preparation results."""

    data: Union[List, np.ndarray, torch.Tensor]
    splits: Dict[str, Any]
    metadata: Dict[str, Any]
    data_type: str  # "dti", "tox21", "properties", "vae"

    @property
    def train_size(self) -> int:
        return len(self.splits.get("train", []))

    @property
    def val_size(self) -> int:
        return len(self.splits.get("val", []))

    @property
    def test_size(self) -> int:
        return len(self.splits.get("test", []))

    @property
    def total_samples(self) -> int:
        return self.metadata.get("total_samples", len(self.data))


class BasePreparer(ABC):
    """Base class for dataset preparers."""

    def __init__(self):
        """Initialize preparer."""
        self.name = self.__class__.__name__

    @abstractmethod
    def prepare(self, **kwargs) -> Tuple[Any, Dict, Dict]:
        """
        Prepare dataset.

        Returns:
            Tuple of (data, splits, metadata)
        """
        pass

    def validate_data(self, data: Any) -> bool:
        """Validate prepared data."""
        if data is None:
            return False
        if hasattr(data, '__len__') and len(data) == 0:
            return False
        return True