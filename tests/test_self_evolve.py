"""Tests for usage-driven self-evolution."""

import pytest

from triz_ai.engine.analyzer import AnalysisResult
from triz_ai.evolution.self_evolve import (
    ConsolidationResult,
    SearchObservation,
    _make_observation_id,
    collect_search_observations,
)
from triz_ai.patents.store import PatentStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    s = PatentStore(db_path=db_path)
    s.init_db()
    yield s
    s.close()


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


def test_collect_search_observations_filters_web_results(store):
    """Only patent_examples with a 'source' field should be collected."""
    result = AnalysisResult(
        problem="reduce thermal resistance in power module",
        method="technical_contradiction",
        improving_param={"id": 17, "name": "Temperature"},
        worsening_param={"id": 14, "name": "Strength"},
        recommended_principles=[
            {"id": 35, "name": "Parameter changes", "description": "..."},
            {"id": 2, "name": "Taking out", "description": "..."},
        ],
        contradiction_confidence=0.85,
        patent_examples=[
            {"id": "US123", "title": "Patent from DB", "abstract": "Local patent"},
            {
                "id": "web1",
                "title": "PCM Thermal Management",
                "abstract": "Phase change materials for cooling",
                "url": "https://example.com/pcm",
                "source": "web_search",
            },
            {
                "id": "web2",
                "title": "Heat Pipe Innovation",
                "abstract": "Novel heat pipe design",
                "url": "https://example.com/heatpipe",
                "source": "web_search",
            },
        ],
    )
    count = collect_search_observations(result, store)
    assert count == 2
    observations = store.get_unconsolidated_observations()
    assert len(observations) == 2
    assert observations[0].improving_param == 17
    assert observations[0].worsening_param == 14
    assert observations[0].principle_ids == [35, 2]
    assert observations[0].analysis_method == "technical_contradiction"


def test_collect_skips_when_no_web_results(store):
    result = AnalysisResult(
        problem="test problem",
        patent_examples=[{"id": "US123", "title": "Local Patent", "abstract": "..."}],
    )
    count = collect_search_observations(result, store)
    assert count == 0
    assert store.get_analyses_since_consolidation() == 0


def test_collect_skips_empty_title(store):
    result = AnalysisResult(
        problem="test",
        patent_examples=[{"title": "", "abstract": "no title", "source": "web_search"}],
    )
    count = collect_search_observations(result, store)
    assert count == 0


def test_collect_increments_analysis_count(store):
    result = AnalysisResult(
        problem="test",
        patent_examples=[{"title": "Web Result", "abstract": "...", "source": "web_search"}],
    )
    collect_search_observations(result, store)
    assert store.get_analyses_since_consolidation() == 1
    collect_search_observations(result, store)
    assert store.get_analyses_since_consolidation() == 2


def test_collect_handles_non_contradiction_methods(store):
    result = AnalysisResult(
        problem="detect cracks in wafer",
        method="su_field",
        patent_examples=[
            {"title": "Ultrasonic Detection", "abstract": "...", "source": "web_search"}
        ],
    )
    count = collect_search_observations(result, store)
    assert count == 1
    obs = store.get_unconsolidated_observations()[0]
    assert obs.improving_param is None
    assert obs.worsening_param is None
    assert obs.analysis_method == "su_field"
