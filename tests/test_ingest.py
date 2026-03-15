"""Tests for patent ingestion."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from triz_ai.llm.client import PatentClassification
from triz_ai.patents.ingest import ingest_directory, ingest_file
from triz_ai.patents.store import PatentStore

FIXTURES = Path(__file__).parent / "fixtures" / "sample_patents"


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    s = PatentStore(db_path=db_path)
    s.init_db()
    yield s
    s.close()


def test_ingest_txt(store):
    patents, classified = ingest_file(
        FIXTURES / "battery_thermal.txt", store, embed=False, auto_classify=False
    )
    assert len(patents) == 1
    assert classified == 0
    assert "Thermal Management" in patents[0].title
    retrieved = store.get_patent(patents[0].id)
    assert retrieved is not None


def test_ingest_json(store):
    patents, classified = ingest_file(
        FIXTURES / "batch_patents.json", store, embed=False, auto_classify=False
    )
    assert len(patents) == 3
    assert classified == 0
    assert patents[0].id == "US2024001234"
    assert patents[0].domain == "battery technology"
    for p in patents:
        assert store.get_patent(p.id) is not None


def test_ingest_directory(store):
    patents, classified = ingest_directory(FIXTURES, store, embed=False, auto_classify=False)
    # 3 txt files + 3 json patents = 6 patents
    assert len(patents) == 6
    assert classified == 0
    all_patents = store.get_all_patents()
    assert len(all_patents) == 6


def test_ingest_unsupported_file(store, tmp_path):
    unsupported = tmp_path / "test.docx"
    unsupported.write_text("test")
    with pytest.raises(ValueError, match="Unsupported"):
        ingest_file(unsupported, store, embed=False, auto_classify=False)


def test_ingest_missing_file(store):
    with pytest.raises(FileNotFoundError):
        ingest_file(Path("/nonexistent/file.txt"), store, embed=False, auto_classify=False)


def test_ingest_with_classify(store):
    """Auto-classification happens during ingestion when enabled."""
    mock_llm = MagicMock()
    mock_llm.classify_patent.return_value = PatentClassification(
        principle_ids=[1, 14],
        contradiction={"improving": 9, "worsening": 1},
        confidence=0.85,
        reasoning="Uses segmentation",
    )
    mock_llm.get_embedding.return_value = None

    patents, classified = ingest_file(
        FIXTURES / "battery_thermal.txt",
        store,
        llm_client=mock_llm,
        embed=False,
        auto_classify=True,
    )
    assert len(patents) == 1
    assert classified == 1
    mock_llm.classify_patent.assert_called_once()

    # Verify classification was stored
    classification = store.get_classification(patents[0].id)
    assert classification is not None
    assert classification.principle_ids == [1, 14]


def test_ingest_classify_failure_does_not_block(store):
    """Classification failure should not block ingestion."""
    mock_llm = MagicMock()
    mock_llm.classify_patent.side_effect = RuntimeError("LLM error")
    mock_llm.get_embedding.return_value = None

    patents, classified = ingest_file(
        FIXTURES / "battery_thermal.txt",
        store,
        llm_client=mock_llm,
        embed=False,
        auto_classify=True,
    )
    assert len(patents) == 1
    assert classified == 0
    # Patent was still ingested despite classification failure
    assert store.get_patent(patents[0].id) is not None
