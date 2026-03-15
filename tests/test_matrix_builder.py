"""Tests for hybrid matrix builder with mocked LLM."""

import json
from unittest.mock import MagicMock, patch

import pytest

from triz_ai.knowledge.contradictions import load_matrix
from triz_ai.knowledge.matrix_builder import seed_matrix
from triz_ai.llm.client import MatrixEntry, MatrixSeedResult


@pytest.fixture
def mock_llm():
    client = MagicMock()
    return client


@pytest.fixture
def tmp_matrix(tmp_path):
    """Create a temporary matrix.json with a few entries and patch _DATA_DIR."""
    matrix_data = {
        "1,2": [15, 8, 29, 34],
        "1,3": [29, 17, 38, 34],
        "2,1": [10, 1, 29, 35],
    }
    matrix_file = tmp_path / "matrix.json"
    matrix_file.write_text(json.dumps(matrix_data))
    return tmp_path, matrix_file


class TestSeedMatrix:
    def test_seed_fills_missing_cells(self, mock_llm, tmp_matrix):
        """seed_matrix() should call LLM and write new entries to matrix.json."""
        data_dir, matrix_file = tmp_matrix

        # Mock LLM to return one entry for each call
        def make_result(improving, worsening_params):
            entries = [
                MatrixEntry(improving=improving, worsening=w, principles=[1, 2])
                for w in worsening_params[:2]  # return 2 entries per call
            ]
            return MatrixSeedResult(entries=entries)

        mock_llm.seed_matrix_row.side_effect = make_result

        with patch("triz_ai.knowledge.matrix_builder._DATA_DIR", data_dir):
            # Clear cache before patching
            load_matrix.cache_clear()
            cells = seed_matrix(mock_llm)

        assert cells > 0
        assert mock_llm.seed_matrix_row.called

        # Verify matrix.json was updated
        with open(matrix_file) as f:
            updated = json.load(f)
        assert len(updated) > 3  # more than original 3 entries

        # Clean up cache
        load_matrix.cache_clear()

    def test_seed_validates_principle_ids(self, mock_llm, tmp_matrix):
        """Entries with invalid principle IDs (>40) should be skipped."""
        data_dir, matrix_file = tmp_matrix

        # Return entry with invalid principle ID 99
        mock_llm.seed_matrix_row.return_value = MatrixSeedResult(
            entries=[
                MatrixEntry(improving=1, worsening=4, principles=[99, 2]),  # invalid
                MatrixEntry(improving=1, worsening=5, principles=[1, 3]),  # valid
            ]
        )

        with patch("triz_ai.knowledge.matrix_builder._DATA_DIR", data_dir):
            load_matrix.cache_clear()
            seed_matrix(mock_llm)

        with open(matrix_file) as f:
            updated = json.load(f)

        # The invalid entry should have been skipped
        assert "1,4" not in updated
        # The valid entry might be present (if 1,5 was in missing set)
        load_matrix.cache_clear()

    def test_seed_validates_max_4_principles(self, mock_llm, tmp_matrix):
        """Entries with more than 4 principles should be skipped."""
        data_dir, matrix_file = tmp_matrix

        mock_llm.seed_matrix_row.return_value = MatrixSeedResult(
            entries=[
                MatrixEntry(improving=1, worsening=4, principles=[1, 2, 3, 4, 5]),  # too many
            ]
        )

        with patch("triz_ai.knowledge.matrix_builder._DATA_DIR", data_dir):
            load_matrix.cache_clear()
            seed_matrix(mock_llm)

        with open(matrix_file) as f:
            updated = json.load(f)

        assert "1,4" not in updated
        load_matrix.cache_clear()

    def test_seed_no_missing_cells(self, mock_llm, tmp_path):
        """When all cells are filled, seed_matrix returns 0."""
        # Create a fully filled 50x50 matrix (all non-diagonal cells)
        matrix_data = {}
        for i in range(1, 51):
            for j in range(1, 51):
                if i != j:
                    matrix_data[f"{i},{j}"] = [1]
        matrix_file = tmp_path / "matrix.json"
        matrix_file.write_text(json.dumps(matrix_data))

        with (
            patch("triz_ai.knowledge.matrix_builder._DATA_DIR", tmp_path),
            patch("triz_ai.knowledge.contradictions._DATA_DIR", tmp_path),
        ):
            load_matrix.cache_clear()
            cells = seed_matrix(mock_llm)

        assert cells == 0
        mock_llm.seed_matrix_row.assert_not_called()
        load_matrix.cache_clear()

    def test_seed_handles_llm_errors(self, mock_llm, tmp_matrix):
        """LLM errors for individual rows should be skipped, not crash."""
        data_dir, matrix_file = tmp_matrix

        mock_llm.seed_matrix_row.side_effect = Exception("LLM error")

        with patch("triz_ai.knowledge.matrix_builder._DATA_DIR", data_dir):
            load_matrix.cache_clear()
            cells = seed_matrix(mock_llm)

        assert cells == 0
        load_matrix.cache_clear()

    def test_seed_clears_cache(self, mock_llm, tmp_matrix):
        """After seeding, load_matrix cache should be cleared."""
        data_dir, _ = tmp_matrix

        mock_llm.seed_matrix_row.return_value = MatrixSeedResult(entries=[])

        with patch("triz_ai.knowledge.matrix_builder._DATA_DIR", data_dir):
            load_matrix.cache_clear()
            # Prime the cache
            load_matrix()

            seed_matrix(mock_llm)

            # Cache should have been cleared (misses reset)
            info_after = load_matrix.cache_info()
            # After clear, hits should be 0
            assert info_after.hits == 0

        load_matrix.cache_clear()
