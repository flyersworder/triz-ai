"""Tests that PatentStore honors EmbeddingsConfig.dimensions.

Regression tests for issue #17: PatentStore was hardcoding 768-d vector
tables regardless of the configured embedding model, silently breaking
every non-768-d embedding (e.g. text-embedding-3-small → 1536).
"""

from __future__ import annotations

import re

import pytest

from triz_ai.patents.store import Patent, PatentStore
from triz_ai.patents.vector import SqliteVecStore


def _schema_dim(store: PatentStore) -> int | None:
    """Read the actual dimension from the patent_embeddings CREATE TABLE SQL."""
    conn = store._get_conn()
    row = conn.execute("SELECT sql FROM sqlite_master WHERE name='patent_embeddings'").fetchone()
    if row is None or row[0] is None:
        return None
    m = re.search(r"FLOAT\[(\d+)\]", row[0])
    return int(m.group(1)) if m else None


def test_init_db_respects_explicit_dimensions(tmp_path):
    """Explicit dimensions= must drive the vec0 schema."""
    db_path = tmp_path / "explicit.db"
    store = PatentStore(db_path=db_path, dimensions=1536)
    store.init_db()
    try:
        assert _schema_dim(store) == 1536
    finally:
        store.close()


def test_init_db_picks_up_embeddings_config_dim(tmp_path, monkeypatch):
    """When dimensions= is omitted, PatentStore must read EmbeddingsConfig.dimensions."""
    from triz_ai import config as cfg_mod

    base = cfg_mod.load_config()
    patched = base.model_copy(deep=True)
    patched.embeddings.dimensions = 1536
    monkeypatch.setattr(cfg_mod, "load_config", lambda *_a, **_kw: patched)

    db_path = tmp_path / "from_config.db"
    store = PatentStore(db_path=db_path)
    store.init_db()
    try:
        assert _schema_dim(store) == 1536
    finally:
        store.close()


def test_search_with_configured_dim_does_not_raise(tmp_path):
    """End-to-end: insert + search at 1536-d must work after init_db."""
    db_path = tmp_path / "search.db"
    store = PatentStore(db_path=db_path, dimensions=1536)
    store.init_db()
    try:
        store.insert_patent(Patent(id="P1", title="x"), embedding=[0.1] * 1536)
        results = store.search_patents([0.1] * 1536, limit=5)
        assert results and results[0][0].id == "P1"
    finally:
        store.close()


def test_init_db_rejects_dim_mismatch_against_existing_schema(tmp_path):
    """Opening an existing 1536-d db with a 768-d config must raise, not silently
    no-op the CREATE TABLE IF NOT EXISTS and produce dimension-mismatch errors later."""
    db_path = tmp_path / "mismatch.db"
    s1 = PatentStore(db_path=db_path, dimensions=1536)
    s1.init_db()
    s1.close()

    s2 = PatentStore(db_path=db_path, dimensions=768)
    with pytest.raises(ValueError, match=r"(?i)dim"):
        s2.init_db()
    s2.close()


def test_init_db_force_rebuilds_at_new_dim(tmp_path):
    """force=True must rebuild the vec0 table at the new dimension."""
    db_path = tmp_path / "force.db"
    s1 = PatentStore(db_path=db_path, dimensions=768)
    s1.init_db()
    s1.close()

    s2 = PatentStore(db_path=db_path, dimensions=1536)
    s2.init_db(force=True)
    try:
        assert _schema_dim(s2) == 1536
    finally:
        s2.close()


def test_explicit_vector_store_is_not_overridden(tmp_path):
    """If the caller passes their own VectorStore, PatentStore must not second-guess
    its dimension — the contract is 'caller knows what they're doing'."""
    db_path = tmp_path / "custom_vs.db"
    custom_vs = SqliteVecStore(db_path=db_path, dimensions=384)
    store = PatentStore(db_path=db_path, vector_store=custom_vs, dimensions=1536)
    store.init_db()
    try:
        # Schema reflects the injected vector store's dim (384), not the
        # PatentStore-level dim (1536) — explicit injection wins.
        assert _schema_dim(store) == 384
    finally:
        store.close()
