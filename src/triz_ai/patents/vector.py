"""Pluggable vector database layer for patent embeddings."""

import logging
import sqlite3
import struct
import threading
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
    """Default vector store using sqlite-vec extension.

    Threading model:
      - If db_path is provided (we own the connection), each thread gets its
        own sqlite3.Connection via threading.local. sqlite_vec is loaded once
        per connection. This is required because sqlite3.Connection is not
        shareable across threads.
      - If a connection is passed in (externally-managed), the caller is
        responsible for thread-safety and we use that connection directly.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        dimensions: int = 768,
        connection: sqlite3.Connection | None = None,
    ):
        self._db_path = Path(db_path) if db_path is not None else None
        self._dimensions = dimensions
        self._shared_conn = connection
        self._owns_conn = connection is None
        self._shared_vec_loaded = False
        # Thread-local state for owned connections; unused when _shared_conn is set.
        self._tls = threading.local()

    def _get_conn(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            # Externally-provided connection; no thread-local wrapping.
            # sqlite_vec must be loaded exactly once on the shared conn.
            if not self._shared_vec_loaded:
                self._load_vec(self._shared_conn)
                self._shared_vec_loaded = True
            return self._shared_conn

        conn = getattr(self._tls, "conn", None)
        if conn is None:
            if self._db_path is None:
                raise ValueError("No db_path or connection provided to SqliteVecStore")
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            # busy_timeout is per-connection; Python <=3.13 defaults to 0, so
            # concurrent writers get `database is locked` immediately without
            # this. Match PatentStore's 5s wait.
            conn.execute("PRAGMA busy_timeout=5000")
            self._load_vec(conn)
            self._tls.conn = conn
        return conn

    @staticmethod
    def _load_vec(conn: sqlite3.Connection) -> None:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

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
        conn.commit()

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
        """Close this thread's connection if we own it.

        Connections held by other threads are not closed here — they will be
        released when those threads exit.
        """
        if not self._owns_conn:
            return
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            conn.close()
            self._tls.conn = None
