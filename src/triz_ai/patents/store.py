"""Patent storage with SQLite + pluggable vector search."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from triz_ai.patents.vector import SqliteVecStore, VectorStore

if TYPE_CHECKING:
    from triz_ai.evolution.self_evolve import SearchObservation

logger = logging.getLogger(__name__)


class Patent(BaseModel):
    """Patent data model."""

    id: str
    title: str
    abstract: str | None = None
    claims: str | None = None
    domain: str | None = None
    filing_date: str | None = None
    assignee: str | None = None
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
    assignee TEXT,
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

CREATE TABLE IF NOT EXISTS search_observations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    snippet TEXT,
    url TEXT,
    source_tool TEXT,
    problem_text TEXT,
    analysis_method TEXT,
    improving_param INTEGER,
    worsening_param INTEGER,
    principle_ids JSON,
    analysis_confidence REAL,
    consolidated BOOLEAN DEFAULT 0,
    observed_at TEXT,
    consolidated_at TEXT
);

CREATE TABLE IF NOT EXISTS self_evolution_meta (
    id INTEGER PRIMARY KEY DEFAULT 1,
    analyses_since_consolidation INTEGER DEFAULT 0,
    last_consolidated_at TEXT
);
"""


class PatentStore:
    """SQLite-backed patent storage with vector search."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        vector_store: VectorStore | None = None,
    ):
        if db_path is None:
            from triz_ai.config import get_db_path

            db_path = get_db_path()
        self.db_path = Path(db_path)
        # Per-thread connections: sqlite3.Connection is not shareable across threads.
        self._tls = threading.local()
        self._vector_store: VectorStore | None = vector_store

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            # Wait briefly for the write lock instead of immediately failing when
            # another thread's connection (e.g. the vector store) holds it.
            conn.execute("PRAGMA busy_timeout=5000")
            self._tls.conn = conn
        return conn

    def init_db(self, force: bool = False) -> None:
        """Create database tables. If force=True, drop and recreate.

        Note: with thread-local connections, force-reset only closes this
        thread's connection before unlinking. Call init_db(force=True) from a
        single-threaded moment; other threads' connections to the same file
        will hold it open until they exit.
        """
        if force and self.db_path.exists() and str(self.db_path) != ":memory:":
            # Close vector store too — if it holds an open connection (likely on
            # the current thread), a dangling fd to the about-to-be-unlinked
            # inode causes `disk I/O error` on the next read through the
            # re-opened patent connection.
            if self._vector_store is not None:
                self._vector_store.close()
            self._close()
            self.db_path.unlink()
            logger.info("Deleted existing database at %s", self.db_path)

        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

        # Initialize vector store — create default SqliteVecStore if none provided.
        # Pass db_path (not connection) so each thread gets its own connection.
        if self._vector_store is None:
            self._vector_store = SqliteVecStore(db_path=self.db_path)
        self._vector_store.init(force=force)
        logger.info("Database initialized at %s", self.db_path)

    def _close(self) -> None:
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            conn.close()
            self._tls.conn = None

    def close(self) -> None:
        """Close this thread's connection and the vector store.

        Other threads' connections are not closed — they will be cleaned up
        when those threads exit.
        """
        if self._vector_store is not None:
            self._vector_store.close()
        self._close()

    # --- Patent CRUD ---

    def insert_patent(self, patent: Patent, embedding: list[float] | None = None) -> None:
        """Insert a patent and optionally its embedding."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO patents "
            "(id, title, abstract, claims, domain, filing_date, assignee, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                patent.id,
                patent.title,
                patent.abstract,
                patent.claims,
                patent.domain,
                patent.filing_date,
                patent.assignee,
                patent.source,
            ),
        )
        # Commit the patent row before touching the vector store. This is
        # required, not precautionary: the patent conn holds SQLite's single
        # writer lock until commit, and the vector store's separate thread-local
        # conn would wait busy_timeout (5s) and then raise `database is locked`.
        # Consequence: on vector-insert failure the patent row is already
        # persisted. INSERT OR REPLACE makes retries idempotent.
        conn.commit()
        if embedding is not None:
            self._insert_embedding(patent.id, embedding)

    def _insert_embedding(self, patent_id: str, embedding: list[float]) -> None:
        """Insert or replace a patent embedding via vector store."""
        if self._vector_store is not None:
            self._vector_store.insert(patent_id, embedding)

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
        if self._vector_store is None:
            return []
        id_distances = self._vector_store.search(query_embedding, limit=limit)
        results = []
        for patent_id, distance in id_distances:
            patent = self.get_patent(patent_id)
            if patent:
                results.append((patent, distance))
        return results

    def search_patents_hybrid(
        self,
        query_embedding: list[float],
        principle_ids: list[int] | None = None,
        improving_param: int | None = None,
        worsening_param: int | None = None,
        limit: int = 5,
    ) -> list[tuple[Patent, float]]:
        """Hybrid patent search combining vector similarity with TRIZ matching.

        Scoring formula:
          hybrid_score = (1 - distance) + principle_bonus + contradiction_bonus
        - principle_bonus: 0.3 per overlapping principle (capped at 0.6)
        - contradiction_bonus: 0.4 exact match; 0.2 partial match

        Fetches more candidates from vector search, scores them, and returns top `limit`.
        """
        if self._vector_store is None:
            return []

        # Fetch more candidates than needed for re-ranking
        candidate_count = max(limit * 4, 20)
        id_distances = self._vector_store.search(query_embedding, limit=candidate_count)

        if not id_distances:
            return []

        principle_ids_set = set(principle_ids or [])
        scored: list[tuple[Patent, float]] = []

        for patent_id, distance in id_distances:
            patent = self.get_patent(patent_id)
            if patent is None:
                continue

            # Base score: vector similarity (1 - distance)
            score = 1.0 - distance

            # Boost from classification match
            if principle_ids_set or improving_param is not None:
                classification = self.get_classification(patent.id)
                if classification:
                    # Principle overlap bonus: 0.3 per overlap, capped at 0.6
                    if principle_ids_set:
                        overlap = len(principle_ids_set & set(classification.principle_ids))
                        score += min(overlap * 0.3, 0.6)

                    # Contradiction match bonus
                    if improving_param is not None or worsening_param is not None:
                        c_imp = classification.contradiction.get("improving")
                        c_wor = classification.contradiction.get("worsening")
                        if c_imp == improving_param and c_wor == worsening_param:
                            score += 0.4
                        elif c_imp == improving_param or c_wor == worsening_param:
                            score += 0.2

            scored.append((patent, score))

        # Sort by hybrid score descending, return top `limit`
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

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
                assignee=row_dict.get("assignee"),
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

    def get_next_candidate_parameter_id(self) -> int:
        """Get the next available candidate parameter ID number."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COALESCE(MAX(CAST(SUBSTR(id, 2) AS INTEGER)), 0) AS max_id "
            "FROM candidate_parameters"
        ).fetchone()
        return row["max_id"] + 1

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

    def get_next_candidate_id(self) -> int:
        """Get the next available candidate principle ID number.

        Uses MAX(id) instead of COUNT(*) to avoid collisions when
        candidates have been deleted.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COALESCE(MAX(CAST(SUBSTR(id, 2) AS INTEGER)), 0) AS max_id "
            "FROM candidate_principles"
        ).fetchone()
        return row["max_id"] + 1

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

    # --- Search Observations (self-evolution) ---

    def insert_search_observation(self, observation: SearchObservation) -> None:
        """Insert a search observation, ignoring duplicates."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO search_observations "
            "(id, title, snippet, url, source_tool, problem_text, analysis_method, "
            "improving_param, worsening_param, principle_ids, analysis_confidence, "
            "consolidated, observed_at, consolidated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                observation.id,
                observation.title,
                observation.snippet,
                observation.url,
                observation.source_tool,
                observation.problem_text,
                observation.analysis_method,
                observation.improving_param,
                observation.worsening_param,
                json.dumps(observation.principle_ids),
                observation.analysis_confidence,
                observation.consolidated,
                observation.observed_at,
                observation.consolidated_at,
            ),
        )
        conn.commit()

    def get_unconsolidated_observations(self) -> list[SearchObservation]:
        """Get all search observations not yet consolidated."""
        from triz_ai.evolution.self_evolve import SearchObservation

        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM search_observations WHERE consolidated = 0 ORDER BY observed_at"
        ).fetchall()
        return [
            SearchObservation(
                id=row["id"],
                title=row["title"],
                snippet=row["snippet"],
                url=row["url"],
                source_tool=row["source_tool"],
                problem_text=row["problem_text"],
                analysis_method=row["analysis_method"],
                improving_param=row["improving_param"],
                worsening_param=row["worsening_param"],
                principle_ids=json.loads(row["principle_ids"]) if row["principle_ids"] else [],
                analysis_confidence=row["analysis_confidence"] or 0.0,
                consolidated=bool(row["consolidated"]),
                observed_at=row["observed_at"],
                consolidated_at=row["consolidated_at"],
            )
            for row in rows
        ]

    def mark_observations_consolidated(self, observation_ids: list[str]) -> None:
        """Mark observations as consolidated, preserving existing consolidated_at if set."""
        if not observation_ids:
            return
        conn = self._get_conn()
        consolidated_at = datetime.now(UTC).isoformat()
        placeholders = ",".join("?" for _ in observation_ids)
        conn.execute(
            f"UPDATE search_observations SET consolidated = 1, "
            f"consolidated_at = COALESCE(consolidated_at, ?) "
            f"WHERE id IN ({placeholders})",
            [consolidated_at, *observation_ids],
        )
        conn.commit()

    def prune_observations(self, retention_days: int = 180) -> int:
        """Delete consolidated observations older than retention period."""
        conn = self._get_conn()
        cutoff = datetime.now(UTC).isoformat()
        cursor = conn.execute(
            "DELETE FROM search_observations "
            "WHERE consolidated = 1 AND consolidated_at IS NOT NULL "
            "AND julianday(?) - julianday(consolidated_at) > ?",
            (cutoff, retention_days),
        )
        conn.commit()
        return cursor.rowcount

    # --- Self-Evolution Meta ---

    def _ensure_meta_row(self) -> None:
        """Ensure the single meta row exists."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO self_evolution_meta "
            "(id, analyses_since_consolidation) VALUES (1, 0)"
        )
        conn.commit()

    def increment_analysis_count(self) -> int:
        """Increment analysis counter, return new value."""
        self._ensure_meta_row()
        conn = self._get_conn()
        conn.execute(
            "UPDATE self_evolution_meta SET analyses_since_consolidation = "
            "analyses_since_consolidation + 1 WHERE id = 1"
        )
        conn.commit()
        row = conn.execute(
            "SELECT analyses_since_consolidation FROM self_evolution_meta WHERE id = 1"
        ).fetchone()
        return row["analyses_since_consolidation"]

    def get_analyses_since_consolidation(self) -> int:
        """Get the number of analyses since last consolidation."""
        self._ensure_meta_row()
        conn = self._get_conn()
        row = conn.execute(
            "SELECT analyses_since_consolidation FROM self_evolution_meta WHERE id = 1"
        ).fetchone()
        return row["analyses_since_consolidation"]

    def reset_analysis_count(self) -> None:
        """Reset analysis counter and update last_consolidated_at."""
        self._ensure_meta_row()
        conn = self._get_conn()
        conn.execute(
            "UPDATE self_evolution_meta SET analyses_since_consolidation = 0, "
            "last_consolidated_at = ? WHERE id = 1",
            (datetime.now(UTC).isoformat(),),
        )
        conn.commit()
