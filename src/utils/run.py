"""Run-based artifact management.

Each invocation creates a timestamped run directory under artifacts/runs/
and updates a `latest` symlink so downstream code can find the most recent run.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def create_run(
    base_dir: str = "artifacts",
    run_name: Optional[str] = None,
    update_latest: bool = True,
) -> Path:
    """Create a timestamped run directory and optionally update ``latest``.

    Args:
        base_dir: Root artifact directory (default ``artifacts``).
        run_name: Optional explicit name; auto-generated as ``YYYYMMDD_HHMMSS``
                  when *None*.
        update_latest: Whether to update the ``latest`` symlink (default
                        ``True``).  Set to ``False`` for components that
                        produce output (e.g., benchmarks) but should not
                        override the most-recent model pointer.

    Returns:
        Path to the created run directory.
    """
    base = Path(base_dir)
    runs_dir = base / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create standard subdirectories
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "reports").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    if update_latest:
        latest_link = base / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        os.symlink(str(run_dir.resolve()), str(latest_link))
        logger.info("Updated symlink: %s -> %s", latest_link, run_dir)
    else:
        logger.info("Skipped latest symlink update (update_latest=False)")

    logger.info("Created run directory: %s", run_dir)
    return run_dir


def resolve_models_dir(
    models_dir: str = "artifacts/models",
) -> str:
    """Resolve the models directory, preferring the latest run if available.

    When the default ``artifacts/models`` is passed, this checks whether a
    ``latest`` symlink exists under ``artifacts`` and, if so, returns
    ``artifacts/latest/models``.  Explicit paths are returned unchanged.

    Args:
        models_dir: Requested models directory (default ``artifacts/models``).

    Returns:
        Path to models directory to use.
    """
    p = Path(models_dir)
    # If caller passed an explicit path (not the default) or the path already
    # contains model sub-directories, use it as-is.
    if models_dir != "artifacts/models" and models_dir != "artifacts/latest/models":
        return models_dir

    # Walk up to find 'artifacts' root, then check for latest symlink
    artifacts_root = _find_artifacts_root(p)
    if artifacts_root is None:
        return models_dir

    latest_link = artifacts_root / "latest"
    if latest_link.exists():
        resolved = str(latest_link / "models")
        logger.info("Resolved models dir via latest symlink: %s", resolved)
        return resolved

    return models_dir


def _find_artifacts_root(path: Path) -> Optional[Path]:
    """Walk up from *path* looking for a directory containing ``runs/``."""
    for parent in [path] + list(path.parents):
        if (parent / "runs").is_dir():
            return parent
    return None
