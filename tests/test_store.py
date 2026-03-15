"""Tests for patent store."""

import pytest

from triz_ai.patents.store import (
    CandidateParameter,
    CandidatePrinciple,
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


@pytest.fixture
def sample_patent():
    return Patent(
        id="US123456",
        title="Test Patent",
        abstract="A test patent about batteries",
        claims="1. A battery comprising...",
        domain="battery technology",
        filing_date="2024-01-01",
        source="curated",
    )


def test_init_db(tmp_path):
    db_path = tmp_path / "test.db"
    store = PatentStore(db_path=db_path)
    store.init_db()
    assert db_path.exists()
    store.close()


def test_init_db_force(tmp_path):
    db_path = tmp_path / "test.db"
    store = PatentStore(db_path=db_path)
    store.init_db()
    store.insert_patent(Patent(id="X", title="X"))
    store.close()

    store2 = PatentStore(db_path=db_path)
    store2.init_db(force=True)
    assert store2.get_patent("X") is None
    store2.close()


def test_insert_and_get_patent(store, sample_patent):
    store.insert_patent(sample_patent)
    retrieved = store.get_patent("US123456")
    assert retrieved is not None
    assert retrieved.title == "Test Patent"
    assert retrieved.domain == "battery technology"


def test_get_nonexistent_patent(store):
    assert store.get_patent("NOPE") is None


def test_get_all_patents(store):
    store.insert_patent(Patent(id="A", title="A"))
    store.insert_patent(Patent(id="B", title="B"))
    patents = store.get_all_patents()
    assert len(patents) == 2


def test_insert_patent_with_embedding(store, sample_patent):
    embedding = [0.1] * 768
    store.insert_patent(sample_patent, embedding=embedding)
    retrieved = store.get_patent("US123456")
    assert retrieved is not None


def test_vector_search(store):
    """Test vector similarity search."""
    p1 = Patent(id="P1", title="Battery patent")
    p2 = Patent(id="P2", title="Motor patent")
    # Create distinct embeddings
    emb1 = [1.0] + [0.0] * 767
    emb2 = [0.0] + [1.0] + [0.0] * 766
    store.insert_patent(p1, embedding=emb1)
    store.insert_patent(p2, embedding=emb2)

    # Search with query close to p1
    query = [0.9] + [0.1] + [0.0] * 766
    results = store.search_patents(query, limit=2)
    assert len(results) == 2
    assert results[0][0].id == "P1"  # closest


def test_classification_crud(store, sample_patent):
    store.insert_patent(sample_patent)
    classification = Classification(
        patent_id="US123456",
        principle_ids=[1, 14, 35],
        contradiction={"improving": 9, "worsening": 1},
        confidence=0.85,
    )
    store.insert_classification(classification)

    retrieved = store.get_classification("US123456")
    assert retrieved is not None
    assert retrieved.principle_ids == [1, 14, 35]
    assert retrieved.confidence == 0.85


def test_get_unclassified_patents(store):
    store.insert_patent(Patent(id="A", title="A"))
    store.insert_patent(Patent(id="B", title="B"))
    store.insert_classification(
        Classification(
            patent_id="A",
            principle_ids=[1],
            contradiction={"improving": 1, "worsening": 2},
            confidence=0.9,
        )
    )
    unclassified = store.get_unclassified_patents()
    assert len(unclassified) == 1
    assert unclassified[0].id == "B"


def test_get_classifications_by_domain(store):
    store.insert_patent(Patent(id="A", title="A", domain="battery"))
    store.insert_patent(Patent(id="B", title="B", domain="motor"))
    store.insert_classification(
        Classification(
            patent_id="A",
            principle_ids=[1],
            contradiction={"improving": 1, "worsening": 2},
            confidence=0.9,
        )
    )
    store.insert_classification(
        Classification(
            patent_id="B",
            principle_ids=[2],
            contradiction={"improving": 3, "worsening": 4},
            confidence=0.8,
        )
    )
    results = store.get_classifications_by_domain("battery")
    assert len(results) == 1
    assert results[0][0].id == "A"


def test_candidate_principle_crud(store):
    candidate = CandidatePrinciple(
        id="C1",
        name="Digital Twin Optimization",
        description="Use digital simulation to optimize physical systems",
        evidence_patent_ids=["US123", "US456", "US789"],
        confidence=0.75,
    )
    store.insert_candidate_principle(candidate)

    pending = store.get_pending_candidates()
    assert len(pending) == 1
    assert pending[0].name == "Digital Twin Optimization"
    assert pending[0].evidence_patent_ids == ["US123", "US456", "US789"]

    store.update_candidate_status("C1", "accepted")
    pending = store.get_pending_candidates()
    assert len(pending) == 0


def test_matrix_observation_crud(store, sample_patent):
    """Test insert and aggregation of matrix observations."""
    store.insert_patent(sample_patent)

    # Insert several observations for the same cell
    for i in range(4):
        store.insert_matrix_observation(
            improving=9,
            worsening=1,
            principle_id=1,
            patent_id=f"PAT{i}",
            confidence=0.8 + i * 0.05,
        )
    # Add a second principle with fewer observations
    store.insert_matrix_observation(
        improving=9, worsening=1, principle_id=14, patent_id="PAT0", confidence=0.9
    )
    store.insert_matrix_observation(
        improving=9, worsening=1, principle_id=14, patent_id="PAT1", confidence=0.85
    )

    # With min_count=3, only principle 1 should appear
    obs = store.get_matrix_observations(min_count=3)
    assert (9, 1) in obs
    entries = obs[(9, 1)]
    assert len(entries) == 1
    assert entries[0][0] == 1  # principle_id
    assert entries[0][1] == 4  # count

    # With min_count=2, both principles should appear
    obs2 = store.get_matrix_observations(min_count=2)
    assert len(obs2[(9, 1)]) == 2

    # Verify ordering: principle 1 (count=4) before principle 14 (count=2)
    assert obs2[(9, 1)][0][0] == 1
    assert obs2[(9, 1)][1][0] == 14


def test_hybrid_search_scoring(store):
    """Test hybrid search combines vector similarity with TRIZ matching."""
    # Create patents with embeddings
    p1 = Patent(id="P1", title="Battery patent", assignee="Tesla Inc")
    p2 = Patent(id="P2", title="Motor patent", assignee="Siemens AG")
    p3 = Patent(id="P3", title="Thermal patent", assignee="Samsung SDI")
    emb1 = [1.0] + [0.0] * 767
    emb2 = [0.3] + [0.7] + [0.0] * 766
    emb3 = [0.3] + [0.7] + [0.0] * 766  # Same vector distance as P2
    store.insert_patent(p1, embedding=emb1)
    store.insert_patent(p2, embedding=emb2)
    store.insert_patent(p3, embedding=emb3)

    # Classify P2 with principles [1, 14] and contradiction (9, 1)
    store.insert_classification(
        Classification(
            patent_id="P2",
            principle_ids=[1, 14],
            contradiction={"improving": 9, "worsening": 1},
            confidence=0.9,
        )
    )
    # Classify P3 with principle [1] and different contradiction
    store.insert_classification(
        Classification(
            patent_id="P3",
            principle_ids=[1],
            contradiction={"improving": 5, "worsening": 3},
            confidence=0.8,
        )
    )

    # Query close to P1 in vector space, but with principle/contradiction matching P2
    query = [0.9] + [0.1] + [0.0] * 766
    results = store.search_patents_hybrid(
        query,
        principle_ids=[1, 14, 35],
        improving_param=9,
        worsening_param=1,
        limit=3,
    )
    assert len(results) == 3
    ids = [p.id for p, _score in results]
    # P2 should be boosted above P3 due to principle overlap (2 vs 1) + contradiction match
    assert "P2" in ids
    assert "P1" in ids

    # Verify scores are reasonable
    scores = {p.id: score for p, score in results}
    # P2 gets principle bonus (min(2*0.3, 0.6) = 0.6) + contradiction bonus (0.4)
    assert scores["P2"] > scores["P3"]


def test_hybrid_search_no_classifications(store):
    """Hybrid search works when no classifications exist (falls back to pure vector)."""
    p1 = Patent(id="P1", title="Battery patent")
    store.insert_patent(p1, embedding=[1.0] + [0.0] * 767)
    results = store.search_patents_hybrid(
        [0.9] + [0.1] + [0.0] * 766,
        principle_ids=[1],
        limit=5,
    )
    assert len(results) == 1
    assert results[0][0].id == "P1"


def test_patent_assignee_field(store):
    """Test that assignee field is stored and retrieved."""
    p = Patent(id="P1", title="Test", assignee="Infineon Technologies AG")
    store.insert_patent(p)
    retrieved = store.get_patent("P1")
    assert retrieved is not None
    assert retrieved.assignee == "Infineon Technologies AG"


def test_patent_assignee_none(store):
    """Test that assignee defaults to None."""
    p = Patent(id="P1", title="Test")
    store.insert_patent(p)
    retrieved = store.get_patent("P1")
    assert retrieved is not None
    assert retrieved.assignee is None


def test_candidate_parameter_crud(store):
    candidate = CandidateParameter(
        id="P1",
        name="Cyber Resilience",
        description="Ability of a system to withstand and recover from cyber attacks",
        evidence_patent_ids=["US111", "US222", "US333"],
        confidence=0.80,
    )
    store.insert_candidate_parameter(candidate)

    pending = store.get_pending_candidate_parameters()
    assert len(pending) == 1
    assert pending[0].name == "Cyber Resilience"
    assert pending[0].evidence_patent_ids == ["US111", "US222", "US333"]

    store.update_candidate_parameter_status("P1", "accepted")
    pending = store.get_pending_candidate_parameters()
    assert len(pending) == 0
