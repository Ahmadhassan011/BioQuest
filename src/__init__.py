"""
BioQuest: Autonomous Agentic Molecule Designer
"""

# Do not import subpackages at package import time to avoid pulling in
# optional runtime dependencies (e.g., `streamlit`) when users only need
# programmatic access to modules. Import subpackages explicitly where needed.
__all__ = ["app", "data", "models", "pipelines", "training", "utils"]
