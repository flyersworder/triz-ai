"""Tests for patent ingestion."""

from pathlib import Path

import pytest

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
    patents = ingest_file(FIXTURES / "battery_thermal.txt", store, embed=False)
    assert len(patents) == 1
    assert "Thermal Management" in patents[0].title
    retrieved = store.get_patent(patents[0].id)
    assert retrieved is not None


def test_ingest_json(store):
    patents = ingest_file(FIXTURES / "batch_patents.json", store, embed=False)
    assert len(patents) == 3
    assert patents[0].id == "US2024001234"
    assert patents[0].domain == "battery technology"
    for p in patents:
        assert store.get_patent(p.id) is not None


def test_ingest_directory(store):
    patents = ingest_directory(FIXTURES, store, embed=False)
    # 3 txt files + 3 json patents = 6 patents
    assert len(patents) == 6
    all_patents = store.get_all_patents()
    assert len(all_patents) == 6


def test_ingest_unsupported_file(store, tmp_path):
    unsupported = tmp_path / "test.docx"
    unsupported.write_text("test")
    with pytest.raises(ValueError, match="Unsupported"):
        ingest_file(unsupported, store, embed=False)


def test_ingest_missing_file(store):
    with pytest.raises(FileNotFoundError):
        ingest_file(Path("/nonexistent/file.txt"), store, embed=False)
