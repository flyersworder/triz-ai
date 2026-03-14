"""Tests for evolution pipeline."""

from unittest.mock import MagicMock

import pytest

from triz_ai.evolution.pipeline import run_evolution
from triz_ai.llm.client import CandidatePrincipleProposal, PatentClassification
from triz_ai.patents.store import Patent, PatentStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    s = PatentStore(db_path=db_path)
    s.init_db()
    yield s
    s.close()


@pytest.fixture
def mock_llm():
    client = MagicMock()
    # classify_patent returns low confidence so patents are "poorly mapped"
    client.classify_patent.return_value = PatentClassification(
        principle_ids=[1],
        contradiction={"improving": 1, "worsening": 2},
        confidence=0.4,
        reasoning="Uncertain classification",
    )
    # cluster_patents returns one cluster with all patents
    client.cluster_patents.return_value = [[0, 1, 2]]
    # propose_candidate_principle returns a proposal
    client.propose_candidate_principle.return_value = CandidatePrincipleProposal(
        name="Digital Twin Optimization",
        description="Using virtual simulations to optimize physical systems",
        how_it_differs="Goes beyond existing principles by incorporating real-time feedback",
        confidence=0.75,
    )
    return client


def test_evolution_pipeline(mock_llm, store):
    # Add unclassified patents
    for i in range(3):
        store.insert_patent(Patent(id=f"P{i}", title=f"Patent {i}", abstract=f"Abstract {i}"))

    candidates = run_evolution(
        llm_client=mock_llm,
        store=store,
        confidence_threshold=0.7,
        min_cluster_size=3,
    )

    assert len(candidates) == 1
    assert candidates[0].name == "Digital Twin Optimization"
    assert len(candidates[0].evidence_patent_ids) == 3

    # Verify stored in DB
    pending = store.get_pending_candidates()
    assert len(pending) == 1


def test_evolution_not_enough_patents(mock_llm, store):
    # Only 1 patent — not enough for clustering
    store.insert_patent(Patent(id="P0", title="Solo Patent", abstract="Alone"))
    candidates = run_evolution(llm_client=mock_llm, store=store, min_cluster_size=3)
    assert len(candidates) == 0


def test_evolution_no_unclassified(mock_llm, store):
    # No patents at all
    candidates = run_evolution(llm_client=mock_llm, store=store)
    assert len(candidates) == 0
