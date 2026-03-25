"""
Logging configuration for BioQuest.

Features:
- Rotating file handlers to prevent log files from growing too large
- Module-specific loggers with independent configuration
- Structured logging with context information
- Performance monitoring (execution time, memory usage)
- Request ID tracking for distributed systems
- Console and file output with different formats
- Log level control from code or environment variables
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Optional


class RequestIDFilter(logging.Filter):
    """Add request ID to log records for tracing."""

    def __init__(self, request_id: Optional[str] = None):
        """Initialize filter with optional request ID."""
        super().__init__()
        self.request_id = request_id or "DEFAULT"

    def filter(self, record):
        """Add request_id to log record."""
        record.request_id = self.request_id
        return True


def setup_logging(
    name: str = "bioquest",
    level: str = "INFO",
    log_dir: str = "logs",
    log_file: str = "bioquest.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    request_id: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
) -> logging.Logger:
    """
    Configure logging for BioQuest with rotating file handler.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
              Can also be set via BIOQUEST_LOG_LEVEL environment variable
        log_dir: Directory for log files (will be created if needed)
        log_file: Log file name
        max_bytes: Maximum size of log file before rotation (default: 10 MB)
        backup_count: Number of backup log files to keep (default: 5)
        request_id: Optional request ID for distributed tracing
        console_output: Whether to output to console
        file_output: Whether to output to file

    Returns:
        Configured logger instance

    Examples:
        >>> logger = setup_logging("my_app", level="DEBUG")
        >>> logger.info("Application started")

        >>> logger = setup_logging(
        ...     name="dti_trainer",
        ...     level="INFO",
        ...     log_dir="training_logs"
        ... )
    """
    # Create logs directory if needed
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Get logger instance
    logger = logging.getLogger(name)

    # Check for environment variable override
    env_level = os.getenv("BIOQUEST_LOG_LEVEL", level).upper()
    log_level = getattr(logging, env_level, logging.INFO)
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add request ID filter if provided
    if request_id:
        request_filter = RequestIDFilter(request_id)
        logger.addFilter(request_filter)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        # Simple format for console (with request ID if available)
        if request_id:
            console_format = (
                "%(asctime)s - [%(request_id)s] - %(name)s - "
                "%(levelname)s - %(message)s"
            )
        else:
            console_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        console_formatter = logging.Formatter(console_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Rotating file handler
    if file_output:
        log_file_path = log_path / log_file

        # Use rotating file handler to prevent files from getting too large
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_file_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(log_level)

        # Detailed format for file (with request ID if available)
        if request_id:
            file_format = (
                "%(asctime)s - [%(request_id)s] - %(name)s - "
                "%(levelname)s - %(filename)s:%(lineno)d - %(message)s"
            )
        else:
            file_format = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(filename)s:%(lineno)d - %(message)s"
            )

        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.info(f"Logging configured: level={env_level}, log_dir={log_dir}")

    return logger


def get_module_logger(
    module_name: str,
    level: str = "INFO",
    log_dir: str = "logs",
) -> logging.Logger:
    """
    Get a logger for a specific module.

    Creates module-specific log file in logs/{module_name}.log

    Args:
        module_name: Name of the module (e.g., 'data', 'training', 'prediction')
        level: Logging level
        log_dir: Base directory for logs

    Returns:
        Configured logger for the module

    Examples:
        >>> logger = get_module_logger("data_preparation")
        >>> logger.info("Data loading started")
    """
    logger = setup_logging(
        name=module_name,
        level=level,
        log_dir=log_dir,
        log_file=f"{module_name}.log",
    )
    return logger


def get_performance_logger(log_dir: str = "logs") -> logging.Logger:
    """
    Get a logger for performance metrics.

    Logs training metrics, timing, memory usage, etc.

    Args:
        log_dir: Directory for performance logs

    Returns:
        Logger for performance metrics

    Examples:
        >>> perf_logger = get_performance_logger()
        >>> perf_logger.info(f"Epoch 1: loss=0.542, time=125.3s")
    """
    return setup_logging(
        name="performance",
        level="INFO",
        log_dir=log_dir,
        log_file="performance.log",
        max_bytes=50 * 1024 * 1024,  # 50 MB for detailed metrics
    )


class PerformanceTimer:
    """Context manager for timing code blocks and logging execution time."""

    def __init__(self, logger: logging.Logger, message: str, level: str = "INFO"):
        """
        Initialize timer.

        Args:
            logger: Logger instance
            message: Message to log (e.g., "Data loading")
            level: Log level (INFO, DEBUG, etc.)
        """
        self.logger = logger
        self.message = message
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.start_time = None

    def __enter__(self):
        """Start timer on context entry."""
        import time

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log elapsed time on context exit."""
        import time

        elapsed = time.time() - self.start_time

        if exc_type:
            self.logger.log(
                self.level, f"{self.message} failed after {elapsed:.2f}s: {exc_val}"
            )
        else:
            self.logger.log(self.level, f"{self.message} completed in {elapsed:.2f}s")

        return False


def setup_root_logging(
    level: str = "INFO",
    log_dir: str = "logs",
) -> None:
    """
    Setup root logger that applies to all modules.

    Call this once at application startup.

    Args:
        level: Default logging level
        log_dir: Directory for log files

    Examples:
        >>> from src.utils.logging_config import setup_root_logging
        >>>
        >>> # In your main.py or __init__.py
        >>> setup_root_logging(level="INFO", log_dir="logs")
        >>>
        >>> # Now all modules use this config
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Get environment variable override
    env_level = os.getenv("BIOQUEST_LOG_LEVEL", level).upper()
    log_level = getattr(logging, env_level, logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    log_file = log_path / "bioquest.log"
    file_handler = logging.handlers.RotatingFileHandler(
        str(log_file),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    root_logger.info(f"Root logging configured: level={env_level}, dir={log_dir}")


# Example usage functions
def example_basic_logging():
    """Example: Basic logging setup."""
    logger = setup_logging("my_app")
    logger.info("Application started")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")


def example_module_logging():
    """Example: Module-specific logging."""
    data_logger = get_module_logger("data")
    training_logger = get_module_logger("training")

    data_logger.info("Loading data...")
    training_logger.info("Starting training...")


def example_performance_timing():
    """Example: Performance timing with context manager."""
    logger = setup_logging("timing_example")
    perf_logger = get_performance_logger()

    with PerformanceTimer(logger, "Data loading") as timer:
        # Simulate work
        import time

        time.sleep(0.1)

    with PerformanceTimer(perf_logger, "Model training", level="INFO") as timer:
        import time

        time.sleep(0.2)


if __name__ == "__main__":
    # Run examples
    print("Running logging examples...\n")
    example_basic_logging()
    print("\n" + "=" * 60 + "\n")
    example_module_logging()
    print("\n" + "=" * 60 + "\n")
    example_performance_timing()
