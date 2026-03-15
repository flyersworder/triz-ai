"""Patent storage with SQLite + sqlite-vec for vector search."""

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Patent(BaseModel):
    """Patent data model."""

    id: str
    title: str
    abstract: str | None = None
    claims: str | None = None
    domain: str | None = None
    filing_date: str | None = None
    source: str = "curated"


class Classification(BaseModel):
    """TRIZ classification result for a patent."""

    patent_id: str
    principle_ids: list[int]
    contradiction: dict  # {"improving": int, "worsening": int}
    confidence: float
    classified_at: str | None = None


class CandidateParameter(BaseModel):
    """Candidate new engineering parameter from evolution pipeline."""

    id: str
    name: str
    description: str | None = None
    evidence_patent_ids: list[str] = []
    confidence: float = 0.0
    status: str = "pending_review"
    created_at: str | None = None


class CandidatePrinciple(BaseModel):
    """Candidate new TRIZ principle from evolution pipeline."""

    id: str
    name: str
    description: str | None = None
    evidence_patent_ids: list[str] = []
    confidence: float = 0.0
    status: str = "pending_review"
    created_at: str | None = None


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS patents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    claims TEXT,
    domain TEXT,
    filing_date TEXT,
    source TEXT
);

CREATE TABLE IF NOT EXISTS classifications (
    patent_id TEXT PRIMARY KEY REFERENCES patents(id),
    principle_ids JSON,
    contradiction JSON,
    confidence REAL,
    classified_at TEXT
);

CREATE TABLE IF NOT EXISTS candidate_parameters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    evidence_patent_ids JSON,
    confidence REAL,
    status TEXT DEFAULT 'pending_review',
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS candidate_principles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    evidence_patent_ids JSON,
    confidence REAL,
    status TEXT DEFAULT 'pending_review',
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS matrix_observations (
    improving_param INTEGER NOT NULL,
    worsening_param INTEGER NOT NULL,
    principle_id INTEGER NOT NULL,
    patent_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    observed_at TEXT,
    PRIMARY KEY (improving_param, worsening_param, principle_id, patent_id)
);
"""

_VEC_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS patent_embeddings USING vec0(
    patent_id TEXT PRIMARY KEY,
    embedding FLOAT[768]
);
"""


class PatentStore:
    """SQLite-backed patent storage with vector search."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            from triz_ai.config import get_db_path

            db_path = get_db_path()
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def init_db(self, force: bool = False) -> None:
        """Create database tables. If force=True, drop and recreate."""
        if force and self.db_path.exists() and str(self.db_path) != ":memory:":
            self._close()
            self.db_path.unlink()
            logger.info("Deleted existing database at %s", self.db_path)

        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)

        # Load sqlite-vec extension and create vector table
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.executescript(_VEC_TABLE_SQL)
        conn.commit()
        logger.info("Database initialized at %s", self.db_path)

    def _close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def close(self) -> None:
        """Close database connection."""
        self._close()

    # --- Patent CRUD ---

    def insert_patent(self, patent: Patent, embedding: list[float] | None = None) -> None:
        """Insert a patent and optionally its embedding."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO patents "
            "(id, title, abstract, claims, domain, filing_date, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                patent.id,
                patent.title,
                patent.abstract,
                patent.claims,
                patent.domain,
                patent.filing_date,
                patent.source,
            ),
        )
        if embedding is not None:
            self._insert_embedding(patent.id, embedding)
        conn.commit()

    def _insert_embedding(self, patent_id: str, embedding: list[float]) -> None:
        """Insert or replace a patent embedding."""
        import struct

        conn = self._get_conn()
        # sqlite-vec expects binary float32 data
        embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)
        conn.execute(
            "INSERT OR REPLACE INTO patent_embeddings (patent_id, embedding) VALUES (?, ?)",
            (patent_id, embedding_bytes),
        )

    def get_patent(self, patent_id: str) -> Patent | None:
        """Get a patent by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM patents WHERE id = ?", (patent_id,)).fetchone()
        if row is None:
            return None
        return Patent(**dict(row))

    def search_patents(
        self, query_embedding: list[float], limit: int = 5
    ) -> list[tuple[Patent, float]]:
        """Search patents by vector similarity. Returns (patent, distance) tuples."""
        import struct

        conn = self._get_conn()
        embedding_bytes = struct.pack(f"{len(query_embedding)}f", *query_embedding)
        rows = conn.execute(
            """
            SELECT p.*, pe.distance
            FROM patent_embeddings pe
            JOIN patents p ON p.id = pe.patent_id
            WHERE pe.embedding MATCH ? AND k = ?
            ORDER BY pe.distance
            """,
            (embedding_bytes, limit),
        ).fetchall()
        results = []
        for row in rows:
            row_dict = dict(row)
            distance = row_dict.pop("distance")
            results.append((Patent(**row_dict), distance))
        return results

    def get_all_patents(self) -> list[Patent]:
        """Get all patents."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM patents").fetchall()
        return [Patent(**dict(row)) for row in rows]

    # --- Classification CRUD ---

    def insert_classification(self, classification: Classification) -> None:
        """Insert or replace a classification."""
        conn = self._get_conn()
        classified_at = classification.classified_at or datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO classifications "
            "(patent_id, principle_ids, contradiction, confidence, classified_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                classification.patent_id,
                json.dumps(classification.principle_ids),
                json.dumps(classification.contradiction),
                classification.confidence,
                classified_at,
            ),
        )
        conn.commit()

    def get_classification(self, patent_id: str) -> Classification | None:
        """Get classification for a patent."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM classifications WHERE patent_id = ?", (patent_id,)
        ).fetchone()
        if row is None:
            return None
        row_dict = dict(row)
        row_dict["principle_ids"] = json.loads(row_dict["principle_ids"])
        row_dict["contradiction"] = json.loads(row_dict["contradiction"])
        return Classification(**row_dict)

    def get_unclassified_patents(self) -> list[Patent]:
        """Get patents that haven't been classified yet."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT p.* FROM patents p "
            "LEFT JOIN classifications c ON p.id = c.patent_id "
            "WHERE c.patent_id IS NULL"
        ).fetchall()
        return [Patent(**dict(row)) for row in rows]

    def get_classifications_by_domain(self, domain: str) -> list[tuple[Patent, Classification]]:
        """Get all classified patents in a domain.

        Matches patents where:
        - domain field contains the search term (case-insensitive), OR
        - domain is NULL but title or abstract contains the search term
        """
        conn = self._get_conn()
        term = domain.lower()
        rows = conn.execute(
            "SELECT p.*, c.principle_ids, c.contradiction, c.confidence, c.classified_at "
            "FROM patents p "
            "JOIN classifications c ON p.id = c.patent_id "
            "WHERE LOWER(p.domain) LIKE '%' || ? || '%' "
            "   OR (p.domain IS NULL AND ("
            "       LOWER(p.title) LIKE '%' || ? || '%' "
            "       OR LOWER(p.abstract) LIKE '%' || ? || '%'))",
            (term, term, term),
        ).fetchall()
        results = []
        for row in rows:
            row_dict = dict(row)
            classification = Classification(
                patent_id=row_dict["id"],
                principle_ids=json.loads(row_dict["principle_ids"]),
                contradiction=json.loads(row_dict["contradiction"]),
                confidence=row_dict["confidence"],
                classified_at=row_dict["classified_at"],
            )
            patent = Patent(
                id=row_dict["id"],
                title=row_dict["title"],
                abstract=row_dict["abstract"],
                claims=row_dict["claims"],
                domain=row_dict["domain"],
                filing_date=row_dict["filing_date"],
                source=row_dict["source"],
            )
            results.append((patent, classification))
        return results

    # --- Candidate Parameters ---

    def insert_candidate_parameter(self, candidate: CandidateParameter) -> None:
        """Insert a candidate parameter."""
        conn = self._get_conn()
        created_at = candidate.created_at or datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO candidate_parameters "
            "(id, name, description, evidence_patent_ids, confidence, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                candidate.id,
                candidate.name,
                candidate.description,
                json.dumps(candidate.evidence_patent_ids),
                candidate.confidence,
                candidate.status,
                created_at,
            ),
        )
        conn.commit()

    def get_pending_candidate_parameters(self) -> list[CandidateParameter]:
        """Get candidate parameters pending review."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM candidate_parameters WHERE status = 'pending_review'"
        ).fetchall()
        return [
            CandidateParameter(
                **{
                    **dict(row),
                    "evidence_patent_ids": json.loads(dict(row)["evidence_patent_ids"]),
                }
            )
            for row in rows
        ]

    def update_candidate_parameter_status(self, candidate_id: str, status: str) -> None:
        """Update status of a candidate parameter."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE candidate_parameters SET status = ? WHERE id = ?",
            (status, candidate_id),
        )
        conn.commit()

    # --- Candidate Principles ---

    def insert_candidate_principle(self, candidate: CandidatePrinciple) -> None:
        """Insert a candidate principle."""
        conn = self._get_conn()
        created_at = candidate.created_at or datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO candidate_principles "
            "(id, name, description, evidence_patent_ids, confidence, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                candidate.id,
                candidate.name,
                candidate.description,
                json.dumps(candidate.evidence_patent_ids),
                candidate.confidence,
                candidate.status,
                created_at,
            ),
        )
        conn.commit()

    def get_pending_candidates(self) -> list[CandidatePrinciple]:
        """Get candidate principles pending review."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM candidate_principles WHERE status = 'pending_review'"
        ).fetchall()
        return [
            CandidatePrinciple(
                **{
                    **dict(row),
                    "evidence_patent_ids": json.loads(dict(row)["evidence_patent_ids"]),
                }
            )
            for row in rows
        ]

    def update_candidate_status(self, candidate_id: str, status: str) -> None:
        """Update status of a candidate principle."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE candidate_principles SET status = ? WHERE id = ?",
            (status, candidate_id),
        )
        conn.commit()

    # --- Matrix Observations ---

    def insert_matrix_observation(
        self,
        improving: int,
        worsening: int,
        principle_id: int,
        patent_id: str,
        confidence: float,
    ) -> None:
        """Insert or replace a matrix observation."""
        conn = self._get_conn()
        observed_at = datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO matrix_observations "
            "(improving_param, worsening_param, principle_id, patent_id, confidence, observed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (improving, worsening, principle_id, patent_id, confidence, observed_at),
        )
        conn.commit()

    def get_matrix_observations(
        self, min_count: int = 3
    ) -> dict[tuple[int, int], list[tuple[int, int, float]]]:
        """Get aggregated matrix observations filtered by minimum count.

        Returns dict mapping (improving_param, worsening_param) to list of
        (principle_id, count, avg_confidence) tuples, sorted by count descending.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT improving_param, worsening_param, principle_id, "
            "COUNT(*) as cnt, AVG(confidence) as avg_conf "
            "FROM matrix_observations "
            "GROUP BY improving_param, worsening_param, principle_id "
            "HAVING cnt >= ? "
            "ORDER BY improving_param, worsening_param, cnt DESC",
            (min_count,),
        ).fetchall()
        result: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
        for row in rows:
            key = (row["improving_param"], row["worsening_param"])
            entry = (row["principle_id"], row["cnt"], row["avg_conf"])
            result.setdefault(key, []).append(entry)
        return result
