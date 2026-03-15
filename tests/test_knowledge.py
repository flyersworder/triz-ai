"""Tests for TRIZ knowledge base — principles, parameters, matrix, and new loaders."""

from unittest.mock import MagicMock

from triz_ai.knowledge.contradictions import load_matrix, lookup, lookup_with_observations
from triz_ai.knowledge.parameters import get_parameter, load_parameters
from triz_ai.knowledge.principles import get_principle, load_principles
from triz_ai.knowledge.separation import get_separation_principle, load_separation_principles
from triz_ai.knowledge.solutions import get_solutions_by_class, load_standard_solutions
from triz_ai.knowledge.trends import get_trend, load_evolution_trends


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
    def test_load_50_parameters(self):
        params = load_parameters()
        assert len(params) == 50

    def test_parameter_ids_1_to_50(self):
        params = load_parameters()
        ids = {p.id for p in params}
        assert ids == set(range(1, 51))

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


class TestLookupWithObservations:
    def test_falls_back_to_static_without_store(self):
        """Without a store, returns same as plain lookup."""
        result = lookup_with_observations(1, 9)
        expected = lookup(1, 9)
        assert result == expected

    def test_falls_back_to_static_with_no_observations(self):
        """With a store that has no observations, returns static results."""
        mock_store = MagicMock()
        mock_store.get_matrix_observations.return_value = {}
        result = lookup_with_observations(1, 9, store=mock_store)
        expected = lookup(1, 9)
        assert result == expected

    def test_merges_static_and_observed(self):
        """Observed principles should be merged with static ones."""
        mock_store = MagicMock()
        # Static lookup for (1, 9) returns some principles
        static = lookup(1, 9)
        assert len(static) > 0  # sanity check

        # Observations: principle 5 seen 10 times, principle static[0] seen 5 times
        mock_store.get_matrix_observations.return_value = {
            (1, 9): [
                (5, 10, 0.9),  # new principle, high count
                (static[0], 5, 0.85),  # existing principle, boosted
            ]
        }
        result = lookup_with_observations(1, 9, store=mock_store)
        assert len(result) <= 4
        # Static[0] should be boosted (count 5 + bonus 2 = 7), but principle 5 has count 10
        assert 5 in result
        assert static[0] in result

    def test_returns_top_4_only(self):
        """Should return at most 4 principles even with many observations."""
        mock_store = MagicMock()
        mock_store.get_matrix_observations.return_value = {
            (40, 41): [
                (1, 10, 0.9),
                (2, 8, 0.8),
                (3, 6, 0.7),
                (4, 4, 0.6),
                (5, 3, 0.5),
            ]
        }
        result = lookup_with_observations(40, 41, store=mock_store)
        assert len(result) <= 4

    def test_handles_store_error_gracefully(self):
        """If store raises, falls back to static."""
        mock_store = MagicMock()
        mock_store.get_matrix_observations.side_effect = Exception("DB error")
        result = lookup_with_observations(1, 9, store=mock_store)
        expected = lookup(1, 9)
        assert result == expected


class TestSeparationPrinciples:
    def test_load_4_principles(self):
        principles = load_separation_principles()
        assert len(principles) == 4

    def test_principle_ids_1_to_4(self):
        principles = load_separation_principles()
        ids = {p.id for p in principles}
        assert ids == {1, 2, 3, 4}

    def test_has_required_fields(self):
        for p in load_separation_principles():
            assert p.id > 0
            assert len(p.name) > 0
            assert len(p.description) > 0
            assert len(p.category) > 0
            assert len(p.techniques) > 0
            assert len(p.examples) > 0

    def test_get_by_id(self):
        p = get_separation_principle(1)
        assert p is not None
        assert "time" in p.category.lower()

    def test_get_invalid_id(self):
        assert get_separation_principle(999) is None


class TestStandardSolutions:
    def test_load_solutions(self):
        solutions = load_standard_solutions()
        assert len(solutions) >= 50  # We have 55 in the file

    def test_has_5_classes(self):
        solutions = load_standard_solutions()
        class_ids = {s.class_id for s in solutions}
        assert class_ids == {1, 2, 3, 4, 5}

    def test_has_required_fields(self):
        for s in load_standard_solutions():
            assert len(s.id) > 0
            assert s.class_id > 0
            assert len(s.class_name) > 0
            assert len(s.name) > 0
            assert len(s.description) > 0

    def test_get_by_class(self):
        class1 = get_solutions_by_class(1)
        assert len(class1) > 0
        assert all(s.class_id == 1 for s in class1)

    def test_get_empty_class(self):
        assert get_solutions_by_class(99) == []


class TestEvolutionTrends:
    def test_load_8_trends(self):
        trends = load_evolution_trends()
        assert len(trends) == 8

    def test_trend_ids_1_to_8(self):
        trends = load_evolution_trends()
        ids = {t.id for t in trends}
        assert ids == set(range(1, 9))

    def test_has_required_fields(self):
        for t in load_evolution_trends():
            assert t.id > 0
            assert len(t.name) > 0
            assert len(t.description) > 0
            assert len(t.stages) >= 2

    def test_stages_have_required_fields(self):
        for t in load_evolution_trends():
            for s in t.stages:
                assert s.stage > 0
                assert len(s.name) > 0
                assert len(s.description) > 0

    def test_get_by_id(self):
        t = get_trend(1)
        assert t is not None
        assert "ideality" in t.name.lower()

    def test_get_invalid_id(self):
        assert get_trend(999) is None
