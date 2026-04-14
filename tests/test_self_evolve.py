"""Tests for usage-driven self-evolution."""

from unittest.mock import MagicMock

import pytest

from triz_ai.engine.analyzer import AnalysisResult
from triz_ai.evolution.self_evolve import (
    ConsolidationResult,
    SearchObservation,
    _make_observation_id,
    collect_search_observations,
    consolidate,
)
from triz_ai.llm.client import (
    ObservationValidation,
    ObservationValidationBatch,
    ValidatedPrinciple,
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


@pytest.fixture
def mock_llm():
    client = MagicMock()
    client.validate_observations.return_value = ObservationValidationBatch(
        validations=[
            ObservationValidation(
                observation_id="ws:aaa",
                validated_principles=[
                    ValidatedPrinciple(principle_id=35, confidence=0.9),
                ],
            ),
            ObservationValidation(
                observation_id="ws:bbb",
                validated_principles=[
                    ValidatedPrinciple(principle_id=35, confidence=0.85),
                ],
            ),
            ObservationValidation(
                observation_id="ws:ccc",
                validated_principles=[
                    ValidatedPrinciple(principle_id=35, confidence=0.8),
                ],
            ),
        ]
    )
    client.cluster_patents.return_value = []
    return client


def _insert_observations(store, count, improving=17, worsening=14, principles=None):
    """Helper to insert N observations with the same contradiction pair."""
    if principles is None:
        principles = [35, 2]
    for i in range(count):
        obs = SearchObservation(
            id=f"ws:{chr(97 + i) * 3}",
            title=f"Web Result {i}",
            snippet=f"Snippet about technique {i}",
            source_tool="web_search",
            problem_text=f"Problem {i}",
            analysis_method="technical_contradiction",
            improving_param=improving,
            worsening_param=worsening,
            principle_ids=principles,
            analysis_confidence=0.8,
            observed_at="2026-04-13T10:00:00+00:00",
        )
        store.insert_search_observation(obs)


def test_consolidate_records_matrix_observations(mock_llm, store):
    _insert_observations(store, 3)
    result = consolidate(mock_llm, store, retention_days=180)
    assert result.observations_processed == 3
    assert result.matrix_observations_added >= 1
    obs = store.get_matrix_observations(min_count=1)
    assert (17, 14) in obs


def test_consolidate_marks_observations_as_consolidated(mock_llm, store):
    _insert_observations(store, 3)
    consolidate(mock_llm, store)
    unconsolidated = store.get_unconsolidated_observations()
    assert len(unconsolidated) == 0


def test_consolidate_with_no_observations(mock_llm, store):
    result = consolidate(mock_llm, store)
    assert result.observations_processed == 0
    mock_llm.validate_observations.assert_not_called()


def test_consolidate_skips_non_contradiction_observations(mock_llm, store):
    obs = SearchObservation(
        id="ws:nocontradiction",
        title="Su-Field Result",
        snippet="Detection method",
        source_tool="web_search",
        analysis_method="su_field",
        observed_at="2026-04-13T10:00:00+00:00",
    )
    store.insert_search_observation(obs)
    result = consolidate(mock_llm, store)
    assert result.observations_processed == 1
    mock_llm.validate_observations.assert_not_called()
    assert len(store.get_unconsolidated_observations()) == 0


def test_consolidate_applies_source_confidence_weight(mock_llm, store):
    _insert_observations(store, 3)
    consolidate(mock_llm, store, source_confidence_weight=0.5)
    obs = store.get_matrix_observations(min_count=1)
    if (17, 14) in obs:
        for _pid, _count, avg_conf in obs[(17, 14)]:
            assert avg_conf <= 0.5
