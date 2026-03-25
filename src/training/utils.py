"""
Training Utilities Module: Helper functions for the training pipeline.
"""

import logging
import json
from typing import Dict, Tuple, Any
from pathlib import Path
import numpy as np
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader


logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """
    Recursively converts numpy types in a dictionary or list to native Python types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    else:
        return obj


def create_data_loaders(
    dataset: Any,
    splits: Dict[str, np.ndarray],
    batch_size: int = 32,
    dataset_type: str = "dti",
) -> Tuple[Any, Any, Any]:
    """
    Create training, validation, and test data loaders.
    """
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    if dataset_type == "dti":
        train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = TorchDataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = TorchDataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

    return train_loader, val_loader, test_loader


def save_training_config(
    config: Dict[str, Any],
    output_path: str,
) -> None:
    """Save training configuration to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(convert_numpy_types(config), f, indent=2)
    logger.info(f"Training config saved to {output_path}")
