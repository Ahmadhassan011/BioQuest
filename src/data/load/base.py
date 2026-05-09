"""
Data loading utilities.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BaseLoader:
    """Base class for data loaders."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader.

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load(self, filename: str) -> Any:
        """
        Load data from file.

        Args:
            filename: Name of file to load

        Returns:
            Loaded data
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        return self._load_file(filepath)

    def _load_file(self, filepath: Path) -> Any:
        """Load file. Override in subclass for specific formats."""
        raise NotImplementedError