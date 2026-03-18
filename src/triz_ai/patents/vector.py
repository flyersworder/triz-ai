"""Pluggable vector database layer for patent embeddings."""

import logging
import sqlite3
import struct
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector storage backends.

    Implementations must provide init, insert, search, and close.
    search returns (id, distance) tuples — the vector store knows nothing about Patent objects.
    """

    def init(self, *, force: bool = False) -> None: ...
    def insert(self, id: str, embedding: list[float]) -> None: ...
    def search(self, query: list[float], limit: int = 5) -> list[tuple[str, float]]: ...
    def close(self) -> None: ...


_VEC_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS patent_embeddings USING vec0(
    patent_id TEXT PRIMARY KEY,
    embedding FLOAT[{dimensions}]
);
"""


class SqliteVecStore:
    """Default vector store using sqlite-vec extension."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        dimensions: int = 768,
        connection: sqlite3.Connection | None = None,
    ):
        self._db_path = Path(db_path) if db_path is not None else None
        self._dimensions = dimensions
        self._conn = connection
        self._owns_conn = connection is None  # Only close if we opened it
        self._vec_loaded = False

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if self._db_path is None:
                raise ValueError("No db_path or connection provided to SqliteVecStore")
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
        if not self._vec_loaded:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._vec_loaded = True
        return self._conn

    def init(self, *, force: bool = False) -> None:
        """Create vector table. If force=True, drop and recreate."""
        conn = self._get_conn()
        if force:
            conn.execute("DROP TABLE IF EXISTS patent_embeddings")
        conn.executescript(_VEC_TABLE_SQL.format(dimensions=self._dimensions))
        conn.commit()

    def insert(self, id: str, embedding: list[float]) -> None:
        """Insert or replace a patent embedding."""
        conn = self._get_conn()
        embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)
        conn.execute(
            "INSERT OR REPLACE INTO patent_embeddings (patent_id, embedding) VALUES (?, ?)",
            (id, embedding_bytes),
        )

    def search(self, query: list[float], limit: int = 5) -> list[tuple[str, float]]:
        """Search by vector similarity. Returns (patent_id, distance) tuples."""
        conn = self._get_conn()
        embedding_bytes = struct.pack(f"{len(query)}f", *query)
        rows = conn.execute(
            """
            SELECT patent_id, distance
            FROM patent_embeddings
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (embedding_bytes, limit),
        ).fetchall()
        return [(row["patent_id"], row["distance"]) for row in rows]

    def close(self) -> None:
        """Close connection if we own it."""
        if self._owns_conn and self._conn is not None:
            self._conn.close()
            self._conn = None
