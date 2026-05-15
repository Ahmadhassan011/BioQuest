"""Tests for model components and configurations."""

import pytest
import torch


class TestModelRegistry:
    """Tests for model registry."""

    def test_registry_has_all_models(self):
        """Test that registry contains all models."""
        from src.models.registry import ModelRegistry

        models = ModelRegistry.list_models()

        assert "gnn_dti" in models
        assert "toxicity" in models
        assert "property" in models
        assert "vae" in models

    def test_registry_get_model_unknown(self):
        """Test that unknown model raises error."""
        from src.models.registry import ModelRegistry

        with pytest.raises(ValueError):
            ModelRegistry.get_model("unknown_model")


class TestToxicityModelOutput:
    """Tests for toxicity model output modes."""

    def test_toxicity_model_has_return_logits_param(self, toxicity_model, sample_tensor_batch):
        """Test toxicity model has return_logits parameter."""
        logits = toxicity_model(sample_tensor_batch, return_logits=True)
        probs = toxicity_model(sample_tensor_batch, return_logits=False)

        assert logits.shape == probs.shape
        assert torch.all(probs >= 0) and torch.all(probs <= 1)

    def test_toxicity_model_logits_positive(self, toxicity_model, sample_tensor_batch):
        """Test that logits can be positive or negative."""
        logits = toxicity_model(sample_tensor_batch, return_logits=True)

        # Logits can be any real number
        assert logits.shape == torch.Size([4, 1])

    def test_toxicity_model_default_output(self, toxicity_model, sample_tensor_batch):
        """Test default output is probabilities."""
        output = toxicity_model(sample_tensor_batch)

        # Default should be sigmoid probabilities (0-1)
        assert torch.all(output >= 0) and torch.all(output <= 1)


class TestClassWeights:
    """Tests for class imbalance handling."""

    def test_pos_weight_calculation(self, class_imbalance_data):
        """Test pos_weight calculation from class distribution."""
        pos_weight = class_imbalance_data["n_neg"] / class_imbalance_data["n_pos"]

        assert pos_weight > 10
        assert pos_weight < 15
        assert abs(pos_weight - 12.3) < 0.5

    def test_class_weights_tensor_creation(self, class_imbalance_data):
        """Test that class weights can be converted to tensor."""
        pos_weight = class_imbalance_data["n_neg"] / class_imbalance_data["n_pos"]
        tensor = torch.tensor([pos_weight])

        assert tensor.shape == torch.Size([1])
        assert tensor.item() > 10

    def test_balanced_weights(self):
        """Test balanced class weights."""
        pos_weight = 1.0
        tensor = torch.tensor([pos_weight])

        assert tensor.item() == 1.0


class TestModelCheckpoint:
    """Tests for model checkpoint saving and loading."""

    def test_create_model_checkpoint_signature(self):
        """Test that create_model_checkpoint accepts model_config."""
        from src.models.registry import create_model_checkpoint
        import inspect

        sig = inspect.signature(create_model_checkpoint)
        params = list(sig.parameters.keys())

        assert "model_config" in params

    def test_model_config_optional(self):
        """Test that model_config is optional."""
        from src.models.registry import create_model_checkpoint
        import inspect

        sig = inspect.signature(create_model_checkpoint)
        param = sig.parameters.get("model_config")

        assert param.default is None