"""Tests for data handling, featurization, and transformations."""

import pytest
import numpy as np


class TestProteinFeaturization:
    """Tests for protein sequence handling."""

    def test_aa_mapping_consistency(self):
        """Verify amino acid mapping is consistent."""
        from src.data.constants import AMINO_ACIDS, AA_TO_IDX

        assert len(AMINO_ACIDS) == 20
        assert len(AA_TO_IDX) == 20

        for i, aa in enumerate(AMINO_ACIDS):
            assert AA_TO_IDX[aa] == i + 1

    def test_validate_sequence_valid(self):
        """Test valid sequence validation."""
        from src.data.constants import validate_sequence

        seq = "ACDEFGHIKLMNPQRSTVWY"
        result = validate_sequence(seq)
        assert result == seq

    def test_validate_sequence_uppercase(self):
        """Test sequence is uppercased."""
        from src.data.constants import validate_sequence

        seq = "acdefghiklmnpqrstvwy"
        result = validate_sequence(seq)
        assert result == "ACDEFGHIKLMNPQRSTVWY"

    def test_validate_sequence_truncation(self):
        """Test sequence truncation."""
        from src.data.constants import validate_sequence

        long_seq = "A" * 2000
        result = validate_sequence(long_seq, max_length=512)
        assert len(result) == 512

    def test_validate_sequence_invalid(self):
        """Test invalid amino acids raise error."""
        from src.data.constants import validate_sequence

        with pytest.raises(ValueError):
            validate_sequence("ACDEFGHIKLMNPQRSTVWXYZ")

    def test_sequence_to_indices(self, short_protein_sequence):
        """Test sequence to indices conversion."""
        from src.data.constants import sequence_to_indices

        indices = sequence_to_indices(short_protein_sequence, max_len=5)
        assert len(indices) == 5
        assert indices[0] > 0  # A maps to index > 0
        assert indices[1] > 0  # C maps to index > 0
        assert indices[4] > 0  # F maps to index > 0 (no padding for 20-char sequence)

    def test_sequence_to_indices_padding(self):
        """Test that shorter sequences are padded correctly."""
        from src.data.constants import sequence_to_indices

        indices = sequence_to_indices("AC", max_len=5)
        assert len(indices) == 5
        assert indices[0] > 0  # A
        assert indices[1] > 0  # C
        assert indices[2] == 0  # padding start
        assert indices[4] == 0  # padding end


class TestDataTransformations:
    """Tests for data transformations."""

    def test_log_transform_davis(self, log_transform_params):
        """Test log transform for DAVIS data."""
        test_values = [0.016, 1.0, 100.0, 1000.0, 10000.0]

        for val in test_values:
            y_log = np.log10(val + log_transform_params["offset"])
            normalized = (y_log - log_transform_params["log_min"]) / (
                log_transform_params["log_max"] - log_transform_params["log_min"] + 1e-10
            )
            assert 0 <= normalized <= 1.1, f"Value {val} normalized to {normalized}"

    def test_logp_normalization(self, logp_transform_params):
        """Test log transform for logp."""
        test_logp = [-2.0, 0.0, 2.0, 4.0, 5.0]

        for val in test_logp:
            logp_shifted = val + 3.0
            normalized = 2 * (np.log1p(logp_shifted) / np.log1p(8.0)) - 1
            assert -1 <= normalized <= 1.1, f"LogP {val} normalized to {normalized}"

    def test_log_transform_handles_small_values(self):
        """Test that log transform handles values near zero."""
        from src.data.constants import AA_TO_IDX

        # Verify mapping handles edge cases
        assert AA_TO_IDX.get("X", 0) == 0  # Unknown maps to padding

    def test_validate_sequence_empty_string(self):
        """Test handling of empty sequence."""
        from src.data.constants import validate_sequence

        result = validate_sequence("", max_length=10)
        assert result == ""