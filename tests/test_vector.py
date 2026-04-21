"""Tests for pluggable vector store layer."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from triz_ai.patents.store import Patent, PatentStore
from triz_ai.patents.vector import SqliteVecStore, VectorStore

# --- Protocol conformance ---


def test_sqlite_vec_store_implements_protocol():
    """SqliteVecStore satisfies VectorStore protocol at runtime."""
    store = SqliteVecStore(db_path=":memory:")
    assert isinstance(store, VectorStore)
    store.close()


def test_mock_satisfies_protocol():
    """A mock with the right methods satisfies VectorStore."""
    mock = MagicMock(spec=SqliteVecStore)
    assert isinstance(mock, VectorStore)


# --- SqliteVecStore unit tests ---


@pytest.fixture
def vec_store(tmp_path):
    db_path = tmp_path / "vec_test.db"
    store = SqliteVecStore(db_path=db_path, dimensions=768)
    store.init()
    yield store
    store.close()


def test_insert_and_search(vec_store):
    """Insert embeddings and retrieve by similarity."""
    vec_store.insert("P1", [1.0] + [0.0] * 767)
    vec_store.insert("P2", [0.0] + [1.0] + [0.0] * 766)

    results = vec_store.search([0.9] + [0.1] + [0.0] * 766, limit=2)
    assert len(results) == 2
    ids = [r[0] for r in results]
    assert ids[0] == "P1"  # closest


def test_search_empty(vec_store):
    """Search on empty store returns empty list."""
    results = vec_store.search([0.5] * 768, limit=5)
    assert results == []


def test_search_limit(vec_store):
    """Search respects the limit parameter."""
    for i in range(10):
        vec_store.insert(f"P{i}", [float(i)] + [0.0] * 767)

    results = vec_store.search([5.0] + [0.0] * 767, limit=3)
    assert len(results) == 3


def test_init_force(tmp_path):
    """init(force=True) drops and recreates the vector table."""
    db_path = tmp_path / "force_test.db"
    store = SqliteVecStore(db_path=db_path, dimensions=768)
    store.init()
    store.insert("P1", [1.0] + [0.0] * 767)

    # Force reinit should clear data
    store.init(force=True)
    results = store.search([1.0] + [0.0] * 767, limit=5)
    assert results == []
    store.close()


def test_close_owned_connection(tmp_path):
    """Close only closes connections the store owns."""
    db_path = tmp_path / "close_test.db"
    store = SqliteVecStore(db_path=db_path, dimensions=768)
    store.init()
    store.close()
    # After close, this thread's connection should be released.
    assert getattr(store._tls, "conn", None) is None


def test_close_shared_connection(tmp_path):
    """Close does NOT close connections provided externally."""
    import sqlite3

    db_path = tmp_path / "shared_test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    store = SqliteVecStore(connection=conn, dimensions=768)
    store.init()
    store.close()
    # Connection should still be usable since we don't own it
    conn.execute("SELECT 1")
    conn.close()


# --- PatentStore delegation tests ---


@pytest.fixture
def mock_vector_store():
    mock = MagicMock()
    mock.init = MagicMock()
    mock.insert = MagicMock()
    mock.search = MagicMock(return_value=[])
    mock.close = MagicMock()
    return mock


def test_patent_store_delegates_init(tmp_path, mock_vector_store):
    """PatentStore.init_db delegates to vector_store.init."""
    db_path = tmp_path / "delegate_init.db"
    store = PatentStore(db_path=db_path, vector_store=mock_vector_store)
    store.init_db()
    mock_vector_store.init.assert_called_once_with(force=False)
    store.close()


def test_patent_store_delegates_init_force(tmp_path, mock_vector_store):
    """PatentStore.init_db(force=True) passes force to vector store."""
    db_path = tmp_path / "delegate_force.db"
    store = PatentStore(db_path=db_path, vector_store=mock_vector_store)
    store.init_db()
    # Now call with force
    store.init_db(force=True)
    mock_vector_store.init.assert_called_with(force=True)
    store.close()


def test_patent_store_delegates_insert(tmp_path, mock_vector_store):
    """insert_patent with embedding delegates to vector_store.insert."""
    db_path = tmp_path / "delegate_insert.db"
    store = PatentStore(db_path=db_path, vector_store=mock_vector_store)
    store.init_db()

    emb = [0.1] * 768
    store.insert_patent(Patent(id="P1", title="Test"), embedding=emb)
    mock_vector_store.insert.assert_called_once_with("P1", emb)
    store.close()


def test_patent_store_delegates_search(tmp_path, mock_vector_store):
    """search_patents delegates to vector_store.search and joins metadata."""
    db_path = tmp_path / "delegate_search.db"
    store = PatentStore(db_path=db_path, vector_store=mock_vector_store)
    store.init_db()

    # Insert a patent (no embedding — mock handles that)
    store.insert_patent(Patent(id="P1", title="Battery Patent"))

    # Mock returns matching ID
    mock_vector_store.search.return_value = [("P1", 0.1)]

    query = [0.5] * 768
    results = store.search_patents(query, limit=5)
    mock_vector_store.search.assert_called_once_with(query, limit=5)
    assert len(results) == 1
    assert results[0][0].id == "P1"
    assert results[0][0].title == "Battery Patent"
    assert results[0][1] == 0.1
    store.close()


def test_patent_store_search_skips_missing_patents(tmp_path, mock_vector_store):
    """search_patents skips IDs that don't exist in relational store."""
    db_path = tmp_path / "delegate_skip.db"
    store = PatentStore(db_path=db_path, vector_store=mock_vector_store)
    store.init_db()

    mock_vector_store.search.return_value = [("MISSING", 0.1)]
    results = store.search_patents([0.5] * 768, limit=5)
    assert results == []
    store.close()


def test_patent_store_delegates_close(tmp_path, mock_vector_store):
    """PatentStore.close also closes vector store."""
    db_path = tmp_path / "delegate_close.db"
    store = PatentStore(db_path=db_path, vector_store=mock_vector_store)
    store.init_db()
    store.close()
    mock_vector_store.close.assert_called_once()


def test_patent_store_no_vector_store_returns_empty(tmp_path):
    """search_patents returns [] when no vector store and init_db not called."""
    db_path = tmp_path / "no_vec.db"
    store = PatentStore(db_path=db_path)
    # Don't call init_db, so _vector_store stays None
    # Manually create relational tables
    conn = store._get_conn()
    from triz_ai.patents.store import _SCHEMA_SQL

    conn.executescript(_SCHEMA_SQL)
    conn.commit()

    results = store.search_patents([0.5] * 768, limit=5)
    assert results == []
    store.close()


def test_patent_store_hybrid_delegates(tmp_path, mock_vector_store):
    """search_patents_hybrid delegates vector part and applies TRIZ scoring."""
    db_path = tmp_path / "delegate_hybrid.db"
    store = PatentStore(db_path=db_path, vector_store=mock_vector_store)
    store.init_db()

    store.insert_patent(Patent(id="P1", title="Battery Patent"))
    mock_vector_store.search.return_value = [("P1", 0.2)]

    results = store.search_patents_hybrid([0.5] * 768, limit=5)
    assert len(results) == 1
    assert results[0][0].id == "P1"
    # Score should be 1.0 - 0.2 = 0.8 (no classification bonus)
    assert abs(results[0][1] - 0.8) < 0.001
    store.close()


# --- Thread-safety regression tests (issue #12) ---


def test_sqlite_vec_store_usable_from_worker_thread(tmp_path):
    """SqliteVecStore initialized on main thread must be usable from a worker.

    Regression for issue #12: the cached sqlite3.Connection was bound to the
    thread that first called _get_conn, so cross-thread reads/writes raised
    sqlite3.ProgrammingError.
    """
    db_path = tmp_path / "vec_threaded.db"
    store = SqliteVecStore(db_path=db_path, dimensions=768)
    store.init()
    store.insert("P1", [1.0] + [0.0] * 767)
    store.insert("P2", [0.0, 1.0] + [0.0] * 766)

    query = [0.9, 0.1] + [0.0] * 766

    with ThreadPoolExecutor(max_workers=1) as pool:
        results = pool.submit(store.search, query, 2).result()

    assert len(results) == 2
    assert results[0][0] == "P1"
    store.close()
