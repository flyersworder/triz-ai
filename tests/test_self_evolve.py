"""Tests for usage-driven self-evolution."""

from triz_ai.evolution.self_evolve import (
    ConsolidationResult,
    SearchObservation,
    _make_observation_id,
)


def test_search_observation_defaults():
    obs = SearchObservation(id="ws:abc123", title="Test Result")
    assert obs.consolidated is False
    assert obs.principle_ids == []
    assert obs.analysis_confidence == 0.0
    assert obs.snippet is None


def test_search_observation_full():
    obs = SearchObservation(
        id="ws:abc123",
        title="Thermal Management via Phase Change",
        snippet="A novel approach using PCMs...",
        url="https://example.com/article",
        source_tool="web_search",
        problem_text="reduce heat in power module",
        analysis_method="technical_contradiction",
        improving_param=17,
        worsening_param=14,
        principle_ids=[35, 2],
        analysis_confidence=0.85,
    )
    assert obs.improving_param == 17
    assert obs.principle_ids == [35, 2]


def test_make_observation_id_deterministic():
    id1 = _make_observation_id("Title A", "Snippet A")
    id2 = _make_observation_id("Title A", "Snippet A")
    assert id1 == id2
    assert id1.startswith("ws:")


def test_make_observation_id_differs_for_different_content():
    id1 = _make_observation_id("Title A", "Snippet A")
    id2 = _make_observation_id("Title B", "Snippet B")
    assert id1 != id2


def test_make_observation_id_handles_none_snippet():
    obs_id = _make_observation_id("Title", None)
    assert obs_id.startswith("ws:")


def test_consolidation_result_defaults():
    result = ConsolidationResult()
    assert result.observations_processed == 0
    assert result.matrix_observations_added == 0
    assert result.candidate_principles_proposed == 0
    assert result.candidate_parameters_proposed == 0
    assert result.observations_pruned == 0
