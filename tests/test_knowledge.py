"""Tests for TRIZ knowledge base — principles, parameters, and contradiction matrix."""

from triz_ai.knowledge.contradictions import load_matrix, lookup
from triz_ai.knowledge.parameters import get_parameter, load_parameters
from triz_ai.knowledge.principles import get_principle, load_principles


class TestPrinciples:
    def test_load_40_principles(self):
        principles = load_principles()
        assert len(principles) == 40

    def test_principle_ids_1_to_40(self):
        principles = load_principles()
        ids = {p.id for p in principles}
        assert ids == set(range(1, 41))

    def test_principle_has_required_fields(self):
        principles = load_principles()
        for p in principles:
            assert p.id > 0
            assert len(p.name) > 0
            assert len(p.description) > 0
            assert len(p.sub_principles) > 0
            assert len(p.keywords) > 0

    def test_get_principle_by_id(self):
        p = get_principle(1)
        assert p is not None
        assert "egment" in p.name  # "Segmentation"

    def test_get_principle_invalid_id(self):
        assert get_principle(999) is None


class TestParameters:
    def test_load_39_parameters(self):
        params = load_parameters()
        assert len(params) == 39

    def test_parameter_ids_1_to_39(self):
        params = load_parameters()
        ids = {p.id for p in params}
        assert ids == set(range(1, 40))

    def test_parameter_has_required_fields(self):
        params = load_parameters()
        for p in params:
            assert p.id > 0
            assert len(p.name) > 0
            assert len(p.description) > 0

    def test_get_parameter_by_id(self):
        p = get_parameter(9)
        assert p is not None
        assert "peed" in p.name  # "Speed"

    def test_get_parameter_invalid_id(self):
        assert get_parameter(999) is None


class TestContradictionMatrix:
    def test_matrix_loads(self):
        matrix = load_matrix()
        assert len(matrix) > 0

    def test_matrix_has_entries(self):
        """Matrix should have many non-empty cells."""
        matrix = load_matrix()
        # The published matrix has roughly 1200+ non-empty cells
        assert len(matrix) > 1000

    def test_diagonal_not_in_matrix(self):
        """Diagonal entries (improving == worsening) should not be in matrix."""
        matrix = load_matrix()
        for i in range(1, 40):
            assert (i, i) not in matrix or matrix[(i, i)] == []

    def test_lookup_known_entry(self):
        """Test a known matrix entry: improving weight (1) vs speed (9)."""
        result = lookup(1, 9)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, int) for x in result)

    def test_lookup_returns_principle_ids(self):
        """All returned IDs should be valid principle IDs (1-40)."""
        matrix = load_matrix()
        for (_i, _w), principles in matrix.items():
            for pid in principles:
                assert 1 <= pid <= 40, f"Invalid principle ID {pid} for ({_i}, {_w})"

    def test_lookup_max_4_principles(self):
        """Each cell should have at most 4 principles."""
        matrix = load_matrix()
        for (_i, _w), principles in matrix.items():
            assert len(principles) <= 4, f"Too many principles for ({_i}, {_w}): {principles}"

    def test_lookup_empty_for_missing(self):
        """Looking up a non-existent entry returns empty list."""
        result = lookup(99, 99)
        assert result == []

    def test_matrix_is_asymmetric(self):
        """Verify matrix is asymmetric: (i,j) != (j,i) for at least some entries."""
        matrix = load_matrix()
        asymmetric_count = 0
        for i in range(1, 40):
            for j in range(1, 40):
                if i != j:
                    forward = matrix.get((i, j), [])
                    reverse = matrix.get((j, i), [])
                    if forward != reverse:
                        asymmetric_count += 1
        assert asymmetric_count > 0, "Matrix appears symmetric — should be asymmetric"
