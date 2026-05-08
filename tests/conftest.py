"""
Pytest configuration and shared fixtures for BioQuest tests.
"""

import pytest
import numpy as np
import torch


@pytest.fixture
def sample_protein_sequence():
    """Sample protein sequence for testing."""
    return "ACDEFGHIKLMNPQRSTVWY" * 10  # 200 chars


@pytest.fixture
def short_protein_sequence():
    """Short protein sequence for quick tests."""
    return "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def log_transform_params():
    """Parameters for DAVIS log transformation."""
    return {"log_min": -5.22, "log_max": 4.0, "offset": 1e-6}


@pytest.fixture
def logp_transform_params():
    """Parameters for logp normalization."""
    return {"logp_min": -3.0, "logp_max": 5.0}


@pytest.fixture
def class_imbalance_data():
    """Class imbalance statistics matching Tox21 dataset."""
    return {"n_neg": 72084, "n_pos": 5862}


@pytest.fixture
def toxicity_model():
    """Create a toxicity model for testing."""
    from src.models.toxicity import ToxicityClassifier
    return ToxicityClassifier(input_dim=264)


@pytest.fixture
def sample_tensor_batch():
    """Sample tensor batch for model testing."""
    return torch.randn(4, 264)