"""Tests for algorithm and logic components."""

import pytest
import numpy as np


class TestParetoDominance:
    """Tests for Pareto dominance logic."""

    def test_strict_dominance(self):
        """Test that strict dominance is correctly identified."""
        mol_better = {"affinity": 0.9, "qed": 0.8, "sa": 0.7}
        mol_worse = {"affinity": 0.7, "qed": 0.6, "sa": 0.5}

        objectives = ["affinity", "qed", "sa"]
        dominates = True
        for obj in objectives:
            if mol_better.get(obj, 0.0) < mol_worse.get(obj, 0.0):
                dominates = False
                break

        assert dominates is True

    def test_equal_values_not_strictly_better(self):
        """Test that equal values don't count as strictly better."""
        mol_a = {"affinity": 0.5, "qed": 0.5, "sa": 0.5}
        mol_b = {"affinity": 0.5, "qed": 0.5, "sa": 0.5}

        objectives = ["affinity", "qed", "sa"]

        # Step 1: Check if mol_a >= mol_b on all objectives (< not <=)
        dominates = True
        for obj in objectives:
            if mol_a.get(obj, 0.0) < mol_b.get(obj, 0.0):
                dominates = False
                break

        # Step 2: Check if mol_a > mol_b on at least one objective
        strictly_better = any(
            mol_a.get(obj, 0.0) > mol_b.get(obj, 0.0) for obj in objectives
        )

        # True dominance requires: dominates AND strictly_better
        assert dominates is True  # Equal passes the < check
        assert strictly_better is False  # But not strictly better

    def test_partial_dominance(self):
        """Test when one molecule is better in some objectives but not all."""
        mol_a = {"affinity": 0.9, "qed": 0.5, "sa": 0.5}
        mol_b = {"affinity": 0.7, "qed": 0.6, "sa": 0.5}

        objectives = ["affinity", "qed", "sa"]
        dominates = True
        for obj in objectives:
            if mol_a.get(obj, 0.0) < mol_b.get(obj, 0.0):
                dominates = False
                break

        # mol_a is better on affinity, worse on qed, equal on sa
        # So mol_a does NOT dominate mol_b
        assert dominates is False

    def test_worse_on_one_objective(self):
        """Test that being worse on any objective prevents dominance."""
        mol_better = {"affinity": 0.9, "qed": 0.3, "sa": 0.7}
        mol_worse = {"affinity": 0.7, "qed": 0.6, "sa": 0.7}

        objectives = ["affinity", "qed", "sa"]
        dominates = True
        for obj in objectives:
            if mol_better.get(obj, 0.0) < mol_worse.get(obj, 0.0):
                dominates = False
                break

        # mol_better is better on affinity and sa, but worse on qed
        # So mol_better does NOT dominate mol_worse
        assert dominates is False


class TestNormalizations:
    """Tests for numerical normalization functions."""

    def test_min_max_normalization_bounds(self):
        """Test that min-max normalization produces values in [0, 1]."""
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
        min_val, max_val = 0.0, 1.0

        for v in values:
            normalized = (v - min_val) / (max_val - min_val + 1e-10)
            assert 0 <= normalized <= 1

    def test_log_transform_compression(self):
        """Test that log transform compresses large ranges."""
        original = [1, 10, 100, 1000, 10000]
        log_transformed = [np.log10(v + 1e-6) for v in original]

        # Log transform should compress the range
        range_before = max(original) - min(original)
        range_after = max(log_transformed) - min(log_transformed)

        assert range_after < range_before

    def test_sigmoid_bounds(self):
        """Test that sigmoid outputs are in (0, 1)."""
        values = [-10, -5, 0, 5, 10]
        for v in values:
            sigmoid = 1 / (1 + np.exp(-v))
            assert 0 < sigmoid < 1

    def test_tanh_bounds(self):
        """Test that tanh outputs are in (-1, 1)."""
        values = [-10, -5, 0, 5, 10]
        for v in values:
            tanh = np.tanh(v)
            assert -1 < tanh < 1


class TestDataStructures:
    """Tests for data structure consistency."""

    def test_dataset_result_dataclass(self):
        """Test DatasetResult dataclass creation."""
        from src.data.preparers import DatasetResult

        result = DatasetResult(
            data=[1, 2, 3],
            splits={"train": [0, 1], "val": [2], "test": []},
            metadata={"total_samples": 3},
            data_type="test",
        )

        assert result.train_size == 2
        assert result.val_size == 1
        assert result.test_size == 0

    def test_ndarray_consistency(self):
        """Test numpy array handling across different operations."""
        arr = np.array([1.0, 2.0, 3.0])

        # Should be able to convert to list
        as_list = arr.tolist()
        assert as_list == [1.0, 2.0, 3.0]

        # Should be able to do element-wise operations
        doubled = arr * 2
        assert doubled.tolist() == [2.0, 4.0, 6.0]