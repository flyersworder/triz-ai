"""Tests for parameter evolution pipeline."""

from unittest.mock import MagicMock

import pytest

from triz_ai.patents.store import (
    Classification,
    Patent,
    PatentStore,
)


@pytest.fixture
def store(tmp_path):
    """Create a temporary patent store."""
    db_path = tmp_path / "test.db"
    s = PatentStore(db_path=db_path)
    s.init_db()
    yield s
    s.close()


def _seed_patents(store: PatentStore, count: int = 5, confidence: float = 0.4) -> list[Patent]:
    """Insert patents with low-confidence classifications."""
    patents = []
    for i in range(count):
        p = Patent(id=f"PAT{i}", title=f"Patent {i}", abstract=f"Abstract for patent {i}")
        store.insert_patent(p)
        store.insert_classification(
            Classification(
                patent_id=p.id,
                principle_ids=[1],
                contradiction={"improving": 1, "worsening": 2},
                confidence=confidence,
            )
        )
        patents.append(p)
    return patents


def test_parameter_evolution_no_patents(store):
    """Pipeline returns empty when no patents exist."""
    from triz_ai.evolution.pipeline import run_parameter_evolution

    mock_llm = MagicMock()
    result = run_parameter_evolution(llm_client=mock_llm, store=store)
    assert result == []


def test_parameter_evolution_not_enough_poorly_mapped(store):
    """Pipeline returns empty when too few poorly-mapped patents."""
    from triz_ai.evolution.pipeline import run_parameter_evolution

    # Insert patents with high confidence (above threshold)
    _seed_patents(store, count=5, confidence=0.9)

    mock_llm = MagicMock()
    result = run_parameter_evolution(llm_client=mock_llm, store=store)
    assert result == []
    mock_llm.cluster_patents.assert_not_called()


def test_parameter_evolution_produces_candidates(store):
    """Pipeline produces candidate parameters from low-confidence patents."""
    from triz_ai.evolution.pipeline import run_parameter_evolution

    _seed_patents(store, count=6, confidence=0.3)

    mock_llm = MagicMock()
    # Cluster returns one cluster of indices [0, 1, 2, 3, 4, 5]
    mock_llm.cluster_patents.return_value = [[0, 1, 2, 3, 4, 5]]

    # Proposal result
    mock_proposal = MagicMock()
    mock_proposal.name = "Cyber Resilience"
    mock_proposal.description = "System ability to withstand cyber attacks"
    mock_proposal.confidence = 0.75
    mock_llm.propose_candidate_parameter.return_value = mock_proposal

    result = run_parameter_evolution(llm_client=mock_llm, store=store, confidence_threshold=0.5)

    assert len(result) == 1
    assert result[0].name == "Cyber Resilience"
    assert result[0].id == "P1"
    assert len(result[0].evidence_patent_ids) == 6

    # Verify stored in DB
    pending = store.get_pending_candidate_parameters()
    assert len(pending) == 1
    assert pending[0].name == "Cyber Resilience"


def test_parameter_evolution_skips_small_clusters(store):
    """Pipeline skips clusters smaller than min_cluster_size."""
    from triz_ai.evolution.pipeline import run_parameter_evolution

    _seed_patents(store, count=4, confidence=0.3)

    mock_llm = MagicMock()
    # Two clusters, both too small
    mock_llm.cluster_patents.return_value = [[0, 1], [2, 3]]

    result = run_parameter_evolution(llm_client=mock_llm, store=store, confidence_threshold=0.5)

    assert result == []
    mock_llm.propose_candidate_parameter.assert_not_called()


def test_parameter_evolution_handles_proposal_failure(store):
    """Pipeline handles LLM proposal failures gracefully."""
    from triz_ai.evolution.pipeline import run_parameter_evolution

    _seed_patents(store, count=4, confidence=0.3)

    mock_llm = MagicMock()
    mock_llm.cluster_patents.return_value = [[0, 1, 2, 3]]
    mock_llm.propose_candidate_parameter.side_effect = Exception("LLM error")

    result = run_parameter_evolution(llm_client=mock_llm, store=store, confidence_threshold=0.5)

    assert result == []
