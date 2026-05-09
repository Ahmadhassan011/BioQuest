"""
Logging configuration for BioQuest.

Provides structured logging setup.
"""

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO, log_file: str = None) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level
        log_file: Optional log file path
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at level {level}")