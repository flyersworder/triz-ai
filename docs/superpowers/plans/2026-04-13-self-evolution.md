# Usage-Driven Self-Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable triz-ai to learn from web search results encountered during every `analyze` call, accumulating observations and periodically consolidating them into refined matrix observations and candidate principles — without user intervention or a patent database.

**Architecture:** Two-phase collect-then-consolidate design. Phase 1 hooks into `route()` / `orchestrate_deep()` to store web search results as `SearchObservation` records after each analysis. Phase 2 is a consolidation pipeline (auto-triggered every N analyses or on-demand via `triz-ai consolidate`) that LLM-validates observations, writes to `matrix_observations`, proposes candidate principles, and prunes old data.

**Tech Stack:** Python 3.12+, SQLite, Pydantic, Typer (CLI), pytest + unittest.mock (testing)

**Spec:** `docs/specs/2026-04-13-self-evolution-design.md`

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `src/triz_ai/evolution/self_evolve.py` | **Create** | `SearchObservation` model, `ConsolidationResult` model, `collect_search_observations()`, `consolidate()`, `maybe_auto_consolidate()`, `_make_observation_id()` |
| `src/triz_ai/patents/repository.py` | Modify | Add 7 new protocol methods for search observations + meta tracking |
| `src/triz_ai/patents/store.py` | Modify | Implement 7 new methods, add 2 new tables to `_SCHEMA_SQL` |
| `src/triz_ai/config.py` | Modify | Add 3 new fields to `EvolutionConfig` |
| `src/triz_ai/engine/router.py` | Modify | Hook `collect_search_observations()` + `maybe_auto_consolidate()` after analysis |
| `src/triz_ai/engine/ariz.py` | Modify | Same hook for deep mode |
| `src/triz_ai/llm/client.py` | Modify | Add `validate_observations()` method |
| `src/triz_ai/llm/prompts.py` | Modify | Add `validate_observations_prompt()` |
| `src/triz_ai/cli.py` | Modify | Add `triz-ai consolidate` command |
| `tests/test_self_evolve.py` | **Create** | Tests for collection, consolidation, auto-trigger |
| `tests/test_store.py` | Modify | Tests for new store methods |
| `tests/test_router.py` | Modify | Test self-evolution hook in `route()` |

---

### Task 1: Data Model — `SearchObservation` and `ConsolidationResult`

**Files:**
- Create: `src/triz_ai/evolution/self_evolve.py`
- Test: `tests/test_self_evolve.py`

- [ ] **Step 1: Write the test for SearchObservation model**

```python
# tests/test_self_evolve.py
"""Tests for usage-driven self-evolution."""

from triz_ai.evolution.self_evolve import (
    ConsolidationResult,
    SearchObservation,
    _make_observation_id,
)


def test_search_observation_defaults():
    obs = SearchObservation(id="ws:abc123", title="Test Result")
    assert obs.consolidated is False
    assert obs.principle_ids == []
    assert obs.analysis_confidence == 0.0
    assert obs.snippet is None


def test_search_observation_full():
    obs = SearchObservation(
        id="ws:abc123",
        title="Thermal Management via Phase Change",
        snippet="A novel approach using PCMs...",
        url="https://example.com/article",
        source_tool="web_search",
        problem_text="reduce heat in power module",
        analysis_method="technical_contradiction",
        improving_param=17,
        worsening_param=14,
        principle_ids=[35, 2],
        analysis_confidence=0.85,
    )
    assert obs.improving_param == 17
    assert obs.principle_ids == [35, 2]


def test_make_observation_id_deterministic():
    id1 = _make_observation_id("Title A", "Snippet A")
    id2 = _make_observation_id("Title A", "Snippet A")
    assert id1 == id2
    assert id1.startswith("ws:")


def test_make_observation_id_differs_for_different_content():
    id1 = _make_observation_id("Title A", "Snippet A")
    id2 = _make_observation_id("Title B", "Snippet B")
    assert id1 != id2


def test_make_observation_id_handles_none_snippet():
    obs_id = _make_observation_id("Title", None)
    assert obs_id.startswith("ws:")


def test_consolidation_result_defaults():
    result = ConsolidationResult()
    assert result.observations_processed == 0
    assert result.matrix_observations_added == 0
    assert result.candidate_principles_proposed == 0
    assert result.candidate_parameters_proposed == 0
    assert result.observations_pruned == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_self_evolve.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'triz_ai.evolution.self_evolve'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/triz_ai/evolution/self_evolve.py
"""Usage-driven self-evolution — learn from web search results during analysis."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel


class SearchObservation(BaseModel):
    """A web search result captured during analysis, with its analysis context."""

    id: str
    title: str
    snippet: str | None = None
    url: str | None = None
    source_tool: str | None = None
    problem_text: str | None = None
    analysis_method: str | None = None
    improving_param: int | None = None
    worsening_param: int | None = None
    principle_ids: list[int] = []
    analysis_confidence: float = 0.0
    consolidated: bool = False
    observed_at: str | None = None
    consolidated_at: str | None = None


class ConsolidationResult(BaseModel):
    """Summary of a consolidation run."""

    observations_processed: int = 0
    matrix_observations_added: int = 0
    candidate_principles_proposed: int = 0
    candidate_parameters_proposed: int = 0
    observations_pruned: int = 0


def _make_observation_id(title: str, snippet: str | None) -> str:
    """Generate a deterministic ID for deduplication."""
    content = f"{title}|{snippet or ''}"
    hash_hex = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"ws:{hash_hex}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_self_evolve.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/triz_ai/evolution/self_evolve.py tests/test_self_evolve.py
git commit -m "feat: add SearchObservation and ConsolidationResult models for self-evolution"
```

---

### Task 2: Config — Add Self-Evolution Settings to `EvolutionConfig`

**Files:**
- Modify: `src/triz_ai/config.py:39-41`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the test for new config fields**

Add to `tests/test_config.py`:

```python
def test_evolution_config_self_evolution_defaults():
    from triz_ai.config import EvolutionConfig

    config = EvolutionConfig()
    assert config.consolidation_interval == 25
    assert config.retention_days == 180
    assert config.source_confidence_weight == 0.6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_evolution_config_self_evolution_defaults -v`
Expected: FAIL — `AttributeError: 'EvolutionConfig' object has no attribute 'consolidation_interval'`

- [ ] **Step 3: Add new fields to EvolutionConfig**

In `src/triz_ai/config.py`, replace the `EvolutionConfig` class:

```python
class EvolutionConfig(BaseModel):
    auto_classify: bool = True
    review_threshold: float = 0.7
    consolidation_interval: int = 25
    retention_days: int = 180
    source_confidence_weight: float = 0.6
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: All config tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/triz_ai/config.py tests/test_config.py
git commit -m "feat: add self-evolution config fields (consolidation_interval, retention_days, source_confidence_weight)"
```

---

### Task 3: Store — New Tables and Protocol Methods

**Files:**
- Modify: `src/triz_ai/patents/repository.py:1-73`
- Modify: `src/triz_ai/patents/store.py:65-113` (schema), plus new methods at end
- Test: `tests/test_store.py`

- [ ] **Step 1: Write tests for new store methods**

Add to `tests/test_store.py`:

```python
from triz_ai.evolution.self_evolve import SearchObservation


def test_insert_and_get_search_observation(store):
    obs = SearchObservation(
        id="ws:abc123",
        title="Phase Change Thermal Management",
        snippet="Using PCMs for heat dissipation",
        url="https://example.com/article",
        source_tool="web_search",
        problem_text="reduce thermal resistance",
        analysis_method="technical_contradiction",
        improving_param=17,
        worsening_param=14,
        principle_ids=[35, 2],
        analysis_confidence=0.85,
        observed_at="2026-04-13T10:00:00+00:00",
    )
    store.insert_search_observation(obs)

    unconsolidated = store.get_unconsolidated_observations()
    assert len(unconsolidated) == 1
    assert unconsolidated[0].id == "ws:abc123"
    assert unconsolidated[0].title == "Phase Change Thermal Management"
    assert unconsolidated[0].principle_ids == [35, 2]
    assert unconsolidated[0].consolidated is False


def test_insert_search_observation_deduplicates(store):
    obs = SearchObservation(id="ws:abc123", title="Same Result", observed_at="2026-04-13T10:00:00")
    store.insert_search_observation(obs)
    store.insert_search_observation(obs)  # duplicate — should be ignored

    unconsolidated = store.get_unconsolidated_observations()
    assert len(unconsolidated) == 1


def test_mark_observations_consolidated(store):
    obs1 = SearchObservation(id="ws:aaa", title="Result A", observed_at="2026-04-13T10:00:00")
    obs2 = SearchObservation(id="ws:bbb", title="Result B", observed_at="2026-04-13T10:00:00")
    store.insert_search_observation(obs1)
    store.insert_search_observation(obs2)

    store.mark_observations_consolidated(["ws:aaa"])

    unconsolidated = store.get_unconsolidated_observations()
    assert len(unconsolidated) == 1
    assert unconsolidated[0].id == "ws:bbb"


def test_prune_observations(store):
    old_obs = SearchObservation(
        id="ws:old",
        title="Old Result",
        consolidated=True,
        observed_at="2020-01-01T00:00:00",
        consolidated_at="2020-01-02T00:00:00",
    )
    new_obs = SearchObservation(
        id="ws:new",
        title="New Result",
        consolidated=True,
        observed_at="2026-04-13T10:00:00",
        consolidated_at="2026-04-13T11:00:00",
    )
    store.insert_search_observation(old_obs)
    store.insert_search_observation(new_obs)

    # Mark both as consolidated (old_obs already is via field, but store uses INSERT)
    store.mark_observations_consolidated(["ws:old", "ws:new"])

    pruned = store.prune_observations(retention_days=180)
    assert pruned == 1  # only the old one

    unconsolidated = store.get_unconsolidated_observations()
    assert len(unconsolidated) == 0  # both consolidated, but new one not pruned


def test_analysis_count_tracking(store):
    assert store.get_analyses_since_consolidation() == 0

    store.increment_analysis_count()
    assert store.get_analyses_since_consolidation() == 1

    store.increment_analysis_count()
    store.increment_analysis_count()
    assert store.get_analyses_since_consolidation() == 3

    store.reset_analysis_count()
    assert store.get_analyses_since_consolidation() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_store.py::test_insert_and_get_search_observation -v`
Expected: FAIL — `ImportError` or `AttributeError`

- [ ] **Step 3: Update PatentRepository protocol**

In `src/triz_ai/patents/repository.py`, add the import and new methods:

```python
# Add to TYPE_CHECKING imports at top:
if TYPE_CHECKING:
    from triz_ai.evolution.self_evolve import SearchObservation
    from triz_ai.patents.store import (
        CandidateParameter,
        CandidatePrinciple,
        Classification,
        Patent,
    )
```

Add these methods to the `PatentRepository` protocol class, after the existing `get_matrix_observations` method:

```python
    # --- Search Observations (self-evolution) ---
    def insert_search_observation(self, observation: SearchObservation) -> None: ...
    def get_unconsolidated_observations(self) -> list[SearchObservation]: ...
    def mark_observations_consolidated(self, observation_ids: list[str]) -> None: ...
    def prune_observations(self, retention_days: int = 180) -> int: ...

    # --- Self-Evolution Meta ---
    def increment_analysis_count(self) -> int: ...
    def get_analyses_since_consolidation(self) -> int: ...
    def reset_analysis_count(self) -> None: ...
```

- [ ] **Step 4: Add new tables to PatentStore schema**

In `src/triz_ai/patents/store.py`, append to `_SCHEMA_SQL` (before the closing `"""`):

```sql
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
    last_consolidated_at TEXT,
    total_observations INTEGER DEFAULT 0
);
```

- [ ] **Step 5: Implement new PatentStore methods**

Add these methods to the `PatentStore` class in `src/triz_ai/patents/store.py`:

```python
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
            "SELECT * FROM search_observations WHERE consolidated = 0 "
            "ORDER BY observed_at"
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
        """Mark observations as consolidated."""
        if not observation_ids:
            return
        conn = self._get_conn()
        consolidated_at = datetime.now(UTC).isoformat()
        placeholders = ",".join("?" for _ in observation_ids)
        conn.execute(
            f"UPDATE search_observations SET consolidated = 1, consolidated_at = ? "
            f"WHERE id IN ({placeholders})",
            [consolidated_at, *observation_ids],
        )
        conn.commit()

    def prune_observations(self, retention_days: int = 180) -> int:
        """Delete consolidated observations older than retention period.

        Returns number of rows deleted.
        """
        conn = self._get_conn()
        cutoff = datetime.now(UTC).isoformat()
        # Use SQLite date arithmetic: observations consolidated more than
        # retention_days ago are pruned.
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
            "(id, analyses_since_consolidation, total_observations) VALUES (1, 0, 0)"
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
```

Also add the `SearchObservation` import at the top of `store.py` under `TYPE_CHECKING`:

```python
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from triz_ai.patents.vector import SqliteVecStore, VectorStore

if TYPE_CHECKING:
    from triz_ai.evolution.self_evolve import SearchObservation
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: All store tests PASS (existing + 5 new)

- [ ] **Step 7: Run type check**

Run: `uvx ty check src/`
Expected: No new type errors

- [ ] **Step 8: Commit**

```bash
git add src/triz_ai/patents/repository.py src/triz_ai/patents/store.py tests/test_store.py
git commit -m "feat: add search_observations and self_evolution_meta tables with PatentRepository protocol methods"
```

---

### Task 4: Collection — `collect_search_observations()`

**Files:**
- Modify: `src/triz_ai/evolution/self_evolve.py`
- Test: `tests/test_self_evolve.py`

- [ ] **Step 1: Write the test for collection**

Add to `tests/test_self_evolve.py`:

```python
import pytest

from triz_ai.engine.analyzer import AnalysisResult
from triz_ai.evolution.self_evolve import (
    ConsolidationResult,
    SearchObservation,
    _make_observation_id,
    collect_search_observations,
)
from triz_ai.patents.store import PatentStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    s = PatentStore(db_path=db_path)
    s.init_db()
    yield s
    s.close()


def test_collect_search_observations_filters_web_results(store):
    """Only patent_examples with a 'source' field should be collected."""
    result = AnalysisResult(
        problem="reduce thermal resistance in power module",
        method="technical_contradiction",
        improving_param={"id": 17, "name": "Temperature"},
        worsening_param={"id": 14, "name": "Strength"},
        recommended_principles=[
            {"id": 35, "name": "Parameter changes", "description": "..."},
            {"id": 2, "name": "Taking out", "description": "..."},
        ],
        contradiction_confidence=0.85,
        patent_examples=[
            # Local DB result — no source field, should be skipped
            {
                "id": "US123",
                "title": "Patent from DB",
                "abstract": "Local patent",
            },
            # Web search result — has source field, should be collected
            {
                "id": "web1",
                "title": "PCM Thermal Management",
                "abstract": "Phase change materials for cooling",
                "url": "https://example.com/pcm",
                "source": "web_search",
            },
            # Another web result
            {
                "id": "web2",
                "title": "Heat Pipe Innovation",
                "abstract": "Novel heat pipe design",
                "url": "https://example.com/heatpipe",
                "source": "web_search",
            },
        ],
    )

    count = collect_search_observations(result, store)
    assert count == 2

    observations = store.get_unconsolidated_observations()
    assert len(observations) == 2
    assert observations[0].improving_param == 17
    assert observations[0].worsening_param == 14
    assert observations[0].principle_ids == [35, 2]
    assert observations[0].analysis_method == "technical_contradiction"


def test_collect_skips_when_no_web_results(store):
    """No observations should be stored if there are no web results."""
    result = AnalysisResult(
        problem="test problem",
        patent_examples=[
            {"id": "US123", "title": "Local Patent", "abstract": "..."},
        ],
    )
    count = collect_search_observations(result, store)
    assert count == 0
    assert store.get_analyses_since_consolidation() == 0


def test_collect_skips_empty_title(store):
    """Web results without a title should be skipped."""
    result = AnalysisResult(
        problem="test",
        patent_examples=[
            {"title": "", "abstract": "no title", "source": "web_search"},
        ],
    )
    count = collect_search_observations(result, store)
    assert count == 0


def test_collect_increments_analysis_count(store):
    """Analysis counter should increment when observations are stored."""
    result = AnalysisResult(
        problem="test",
        patent_examples=[
            {"title": "Web Result", "abstract": "...", "source": "web_search"},
        ],
    )
    collect_search_observations(result, store)
    assert store.get_analyses_since_consolidation() == 1

    collect_search_observations(result, store)
    # Same observation is deduplicated, but counter still increments
    assert store.get_analyses_since_consolidation() == 2


def test_collect_handles_non_contradiction_methods(store):
    """Methods without improving/worsening params should still collect."""
    result = AnalysisResult(
        problem="detect cracks in wafer",
        method="su_field",
        # No improving_param or worsening_param
        patent_examples=[
            {"title": "Ultrasonic Detection", "abstract": "...", "source": "web_search"},
        ],
    )
    count = collect_search_observations(result, store)
    assert count == 1

    obs = store.get_unconsolidated_observations()[0]
    assert obs.improving_param is None
    assert obs.worsening_param is None
    assert obs.analysis_method == "su_field"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_self_evolve.py::test_collect_search_observations_filters_web_results -v`
Expected: FAIL — `ImportError: cannot import name 'collect_search_observations'`

- [ ] **Step 3: Implement `collect_search_observations()`**

Add to `src/triz_ai/evolution/self_evolve.py`:

```python
import logging
from datetime import UTC, datetime

# Add to existing imports at top, update hashlib import location
```

Then add the function:

```python
logger = logging.getLogger(__name__)


def collect_search_observations(
    result: AnalysisResult,
    store: PatentRepository,
) -> int:
    """Store web search results from analysis as search observations.

    Filters patent_examples to those with a 'source' field (web results),
    builds a SearchObservation from each, and stores in the DB.

    Returns number of observations stored.
    """
    count = 0
    for example in result.patent_examples:
        source = example.get("source")
        if not source:
            continue

        title = example.get("title", "")
        snippet = example.get("abstract", "")
        if not title:
            continue

        obs = SearchObservation(
            id=_make_observation_id(title, snippet),
            title=title,
            snippet=snippet,
            url=example.get("url"),
            source_tool=source,
            problem_text=result.problem,
            analysis_method=result.method,
            improving_param=(
                result.improving_param["id"] if result.improving_param else None
            ),
            worsening_param=(
                result.worsening_param["id"] if result.worsening_param else None
            ),
            principle_ids=[p["id"] for p in result.recommended_principles],
            analysis_confidence=result.contradiction_confidence,
            observed_at=datetime.now(UTC).isoformat(),
        )
        store.insert_search_observation(obs)
        count += 1

    if count > 0:
        store.increment_analysis_count()
        logger.debug("Collected %d search observations", count)

    return count
```

Add the necessary imports at the top of the file:

```python
from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from triz_ai.engine.analyzer import AnalysisResult
    from triz_ai.patents.repository import PatentRepository
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_self_evolve.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/triz_ai/evolution/self_evolve.py tests/test_self_evolve.py
git commit -m "feat: implement collect_search_observations for self-evolution collection phase"
```

---

### Task 5: Router + ARIZ Hooks — Wire Collection into Analysis Pipeline

**Files:**
- Modify: `src/triz_ai/engine/router.py:128-141`
- Modify: `src/triz_ai/engine/ariz.py:309-315`
- Test: `tests/test_router.py`

- [ ] **Step 1: Write the test for collection hook in router**

Add to `tests/test_router.py`:

```python
class TestSelfEvolutionHook:
    """Self-evolution collection hooks in route()."""

    def test_collects_web_results_after_analysis(self, mock_llm, store):
        """route() should call collect_search_observations when research_tools present."""
        from triz_ai.tools import ResearchTool

        def mock_search(query, context):
            return [
                {"title": "Web Result 1", "abstract": "Found via search", "source": "test_tool"},
            ]

        tool = ResearchTool(
            name="test_tool",
            description="Test search",
            fn=mock_search,
            stages=["search"],
        )

        # Patch the pipeline to return a result with web-sourced patent_examples
        mock_result = AnalysisResult(
            problem="test problem",
            method="technical_contradiction",
            improving_param={"id": 1, "name": "Weight"},
            worsening_param={"id": 2, "name": "Length"},
            recommended_principles=[{"id": 10, "name": "Prior action", "description": "..."}],
            patent_examples=[
                {"title": "Web Result 1", "abstract": "Found via search", "source": "test_tool"},
            ],
        )

        with patch(
            "triz_ai.engine.router._get_pipeline",
            return_value=MagicMock(return_value=mock_result),
        ):
            route("test problem", mock_llm, store, research_tools=[tool])

        # Verify observation was stored
        observations = store.get_unconsolidated_observations()
        assert len(observations) == 1
        assert observations[0].source_tool == "test_tool"

    def test_no_collection_without_research_tools(self, mock_llm, store):
        """route() without research_tools should not collect anything."""
        with patch(
            "triz_ai.engine.router._get_pipeline",
            return_value=MagicMock(return_value=AnalysisResult(problem="test")),
        ):
            route("test", mock_llm, store)

        observations = store.get_unconsolidated_observations()
        assert len(observations) == 0

    def test_collection_failure_does_not_break_analysis(self, mock_llm, store):
        """If collection fails, route() should still return the analysis result."""
        from triz_ai.tools import ResearchTool

        tool = ResearchTool(name="t", description="t", fn=lambda q, c: [], stages=["search"])

        mock_result = AnalysisResult(problem="test", method="technical_contradiction")

        with (
            patch(
                "triz_ai.engine.router._get_pipeline",
                return_value=MagicMock(return_value=mock_result),
            ),
            patch(
                "triz_ai.engine.router.collect_search_observations",
                side_effect=Exception("DB error"),
            ),
        ):
            result = route("test", mock_llm, store, research_tools=[tool])
            assert result.problem == "test"  # analysis still returned
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_router.py::TestSelfEvolutionHook -v`
Expected: FAIL — `collect_search_observations` not called / not imported in router

- [ ] **Step 3: Hook collection into `route()`**

In `src/triz_ai/engine/router.py`, after line 131 (the `result = pipeline(...)` call) and before the metadata attachment section, add:

```python
    # Self-evolution: collect web search observations
    if store is not None and research_tools:
        try:
            from triz_ai.evolution.self_evolve import (
                collect_search_observations,
                maybe_auto_consolidate,
            )

            collect_search_observations(result, store)
            maybe_auto_consolidate(llm_client, store)
        except Exception:
            logger.warning("Self-evolution collection failed, continuing")
```

- [ ] **Step 4: Hook collection into `orchestrate_deep()`**

In `src/triz_ai/engine/ariz.py`, before the `return DeepAnalysisResult(...)` statement (around line 309), add:

```python
    # Self-evolution: collect web search observations from all tool results
    if store is not None and research_tools:
        try:
            from triz_ai.evolution.self_evolve import (
                collect_search_observations,
                maybe_auto_consolidate,
            )

            for tool_result in tool_results:
                collect_search_observations(tool_result, store)
            maybe_auto_consolidate(llm_client, store)
        except Exception:
            logger.warning("Self-evolution collection failed, continuing")
```

- [ ] **Step 5: Add `maybe_auto_consolidate` stub**

In `src/triz_ai/evolution/self_evolve.py`, add a stub that will be fully implemented in Task 7:

```python
def maybe_auto_consolidate(
    llm_client: LLMClient,
    store: PatentRepository,
) -> ConsolidationResult | None:
    """Auto-consolidate if analysis count exceeds threshold.

    Returns ConsolidationResult if consolidation ran, None otherwise.
    """
    from triz_ai.config import load_config

    config = load_config()
    count = store.get_analyses_since_consolidation()
    if count < config.evolution.consolidation_interval:
        return None

    result = consolidate(llm_client, store)
    store.reset_analysis_count()
    return result


def consolidate(
    llm_client: LLMClient,
    store: PatentRepository,
    retention_days: int | None = None,
) -> ConsolidationResult:
    """Consolidate search observations into matrix observations and candidates.

    Full implementation in Task 7.
    """
    if retention_days is None:
        from triz_ai.config import load_config

        retention_days = load_config().evolution.retention_days

    observations = store.get_unconsolidated_observations()
    if not observations:
        return ConsolidationResult()

    # Stub: mark as consolidated and prune, no LLM processing yet
    store.mark_observations_consolidated([o.id for o in observations])
    pruned = store.prune_observations(retention_days=retention_days)
    return ConsolidationResult(
        observations_processed=len(observations),
        observations_pruned=pruned,
    )
```

Add the `LLMClient` type hint import:

```python
if TYPE_CHECKING:
    from triz_ai.engine.analyzer import AnalysisResult
    from triz_ai.llm.client import LLMClient
    from triz_ai.patents.repository import PatentRepository
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_router.py -v`
Expected: All router tests PASS (existing + 3 new)

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS — no regressions

- [ ] **Step 8: Commit**

```bash
git add src/triz_ai/engine/router.py src/triz_ai/engine/ariz.py src/triz_ai/evolution/self_evolve.py tests/test_router.py
git commit -m "feat: wire self-evolution collection into route() and orchestrate_deep()"
```

---

### Task 6: LLM — `validate_observations()` Method and Prompt

**Files:**
- Modify: `src/triz_ai/llm/prompts.py`
- Modify: `src/triz_ai/llm/client.py`
- Test: `tests/test_llm_client.py`

- [ ] **Step 1: Write the test for validate_observations**

Add to `tests/test_llm_client.py`:

```python
def test_validate_observations_returns_validated_results(mock_client):
    """validate_observations should return a list of validated principle assignments."""
    from triz_ai.llm.client import ObservationValidation, ObservationValidationBatch

    # Mock the LLM response
    mock_client._complete = MagicMock(
        return_value=ObservationValidationBatch(
            validations=[
                ObservationValidation(
                    observation_id="ws:abc123",
                    validated_principles=[
                        {"principle_id": 35, "confidence": 0.8},
                        {"principle_id": 2, "confidence": 0.3},
                    ],
                ),
            ]
        )
    )

    results = mock_client.validate_observations(
        observations=[
            {
                "id": "ws:abc123",
                "title": "PCM Thermal Management",
                "snippet": "Phase change materials for cooling",
            }
        ],
        improving_param=17,
        improving_name="Temperature",
        worsening_param=14,
        worsening_name="Strength",
        principle_ids=[35, 2],
    )

    assert len(results.validations) == 1
    assert results.validations[0].observation_id == "ws:abc123"
    assert len(results.validations[0].validated_principles) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm_client.py::test_validate_observations_returns_validated_results -v`
Expected: FAIL — `ImportError: cannot import name 'ObservationValidation'`

- [ ] **Step 3: Add the prompt**

Add to `src/triz_ai/llm/prompts.py`:

```python
def validate_observations_prompt() -> str:
    """System prompt for validating search observation principle assignments."""
    principles = _principles_compact()
    return (
        "You are a TRIZ methodology expert. You are given web search results that were "
        "encountered during a TRIZ contradiction analysis. Each result was found when "
        "analyzing a technical contradiction between two engineering parameters.\n\n"
        "Your task: for each search result, validate whether it genuinely supports "
        "the TRIZ principles that were recommended during the analysis. Rate your "
        "confidence (0.0-1.0) for each principle.\n\n"
        "TRIZ Principles reference:\n"
        f"{principles}\n\n"
        "Respond with JSON:\n"
        '{"validations": [{"observation_id": "<id>", "validated_principles": '
        '[{"principle_id": <int>, "confidence": <float 0.0-1.0>}, ...]}, ...]}'
    )
```

- [ ] **Step 4: Add the response models and client method**

Add to `src/triz_ai/llm/client.py`, after the existing response models:

```python
class ValidatedPrinciple(BaseModel):
    principle_id: int
    confidence: float


class ObservationValidation(BaseModel):
    observation_id: str
    validated_principles: list[ValidatedPrinciple]


class ObservationValidationBatch(BaseModel):
    validations: list[ObservationValidation]
```

Add the method to the `LLMClient` class:

```python
    def validate_observations(
        self,
        observations: list[dict],
        improving_param: int,
        improving_name: str,
        worsening_param: int,
        worsening_name: str,
        principle_ids: list[int],
    ) -> ObservationValidationBatch:
        """Validate whether search observations support recommended principles.

        Args:
            observations: List of dicts with id, title, snippet.
            improving_param: The improving parameter ID.
            improving_name: The improving parameter name.
            worsening_param: The worsening parameter ID.
            worsening_name: The worsening parameter name.
            principle_ids: List of principle IDs to validate against.
        """
        from triz_ai.knowledge.parameters import get_parameter
        from triz_ai.knowledge.principles import load_principles

        all_principles = {p.id: p.name for p in load_principles()}
        principle_names = [
            f"{pid}: {all_principles.get(pid, 'Unknown')}" for pid in principle_ids
        ]

        obs_text = "\n---\n".join(
            f"ID: {o['id']}\nTitle: {o['title']}\nSnippet: {o.get('snippet', 'N/A')}"
            for o in observations
        )

        user_prompt = (
            f"Contradiction: improving '{improving_name}' (param {improving_param}) "
            f"worsens '{worsening_name}' (param {worsening_param}).\n\n"
            f"Recommended principles: {', '.join(principle_names)}\n\n"
            f"Search results to validate:\n{obs_text}"
        )

        return self._complete(
            validate_observations_prompt(),
            user_prompt,
            ObservationValidationBatch,
            max_tokens=1024,
        )
```

Add the import of `validate_observations_prompt` to the imports at the top of `client.py`.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_llm_client.py::test_validate_observations_returns_validated_results -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/triz_ai/llm/prompts.py src/triz_ai/llm/client.py tests/test_llm_client.py
git commit -m "feat: add validate_observations LLM method for consolidation"
```

---

### Task 7: Consolidation — Full `consolidate()` Implementation

**Files:**
- Modify: `src/triz_ai/evolution/self_evolve.py`
- Test: `tests/test_self_evolve.py`

- [ ] **Step 1: Write tests for consolidation**

Add to `tests/test_self_evolve.py`:

```python
from collections import defaultdict
from unittest.mock import MagicMock

from triz_ai.evolution.self_evolve import consolidate
from triz_ai.llm.client import ObservationValidation, ObservationValidationBatch, ValidatedPrinciple


@pytest.fixture
def mock_llm():
    client = MagicMock()
    # Default: validate_observations returns high-confidence validations
    client.validate_observations.return_value = ObservationValidationBatch(
        validations=[
            ObservationValidation(
                observation_id="ws:aaa",
                validated_principles=[
                    ValidatedPrinciple(principle_id=35, confidence=0.9),
                ],
            ),
            ObservationValidation(
                observation_id="ws:bbb",
                validated_principles=[
                    ValidatedPrinciple(principle_id=35, confidence=0.85),
                ],
            ),
            ObservationValidation(
                observation_id="ws:ccc",
                validated_principles=[
                    ValidatedPrinciple(principle_id=35, confidence=0.8),
                ],
            ),
        ]
    )
    # cluster_patents for candidate discovery
    client.cluster_patents.return_value = []
    return client


def _insert_observations(store, count, improving=17, worsening=14, principles=None):
    """Helper to insert N observations with the same contradiction pair."""
    if principles is None:
        principles = [35, 2]
    for i in range(count):
        obs = SearchObservation(
            id=f"ws:{chr(97 + i) * 3}",  # ws:aaa, ws:bbb, ws:ccc, ...
            title=f"Web Result {i}",
            snippet=f"Snippet about technique {i}",
            source_tool="web_search",
            problem_text=f"Problem {i}",
            analysis_method="technical_contradiction",
            improving_param=improving,
            worsening_param=worsening,
            principle_ids=principles,
            analysis_confidence=0.8,
            observed_at="2026-04-13T10:00:00+00:00",
        )
        store.insert_search_observation(obs)


def test_consolidate_records_matrix_observations(mock_llm, store):
    """Consolidation should record matrix observations for validated principles."""
    _insert_observations(store, 3)

    result = consolidate(mock_llm, store, retention_days=180)

    assert result.observations_processed == 3
    assert result.matrix_observations_added >= 1

    # Verify matrix observations were recorded
    obs = store.get_matrix_observations(min_count=1)
    assert (17, 14) in obs


def test_consolidate_marks_observations_as_consolidated(mock_llm, store):
    """All processed observations should be marked as consolidated."""
    _insert_observations(store, 3)

    consolidate(mock_llm, store)

    unconsolidated = store.get_unconsolidated_observations()
    assert len(unconsolidated) == 0


def test_consolidate_with_no_observations(mock_llm, store):
    """Consolidation with no observations should return zero result."""
    result = consolidate(mock_llm, store)
    assert result.observations_processed == 0
    mock_llm.validate_observations.assert_not_called()


def test_consolidate_skips_non_contradiction_observations(mock_llm, store):
    """Observations without contradiction params skip matrix recording but still consolidate."""
    obs = SearchObservation(
        id="ws:nocontradiction",
        title="Su-Field Result",
        snippet="Detection method",
        source_tool="web_search",
        analysis_method="su_field",
        # No improving_param or worsening_param
        observed_at="2026-04-13T10:00:00+00:00",
    )
    store.insert_search_observation(obs)

    result = consolidate(mock_llm, store)

    assert result.observations_processed == 1
    # No matrix observations for non-contradiction methods
    mock_llm.validate_observations.assert_not_called()
    # But observation is still marked consolidated
    assert len(store.get_unconsolidated_observations()) == 0


def test_consolidate_applies_source_confidence_weight(mock_llm, store):
    """Matrix observation confidence should be discounted by source_confidence_weight."""
    _insert_observations(store, 3)

    consolidate(mock_llm, store, source_confidence_weight=0.5)

    obs = store.get_matrix_observations(min_count=1)
    if (17, 14) in obs:
        for _pid, _count, avg_conf in obs[(17, 14)]:
            assert avg_conf <= 0.5  # weighted down from 0.8-0.9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_self_evolve.py::test_consolidate_records_matrix_observations -v`
Expected: FAIL — consolidate stub doesn't do LLM validation

- [ ] **Step 3: Replace the `consolidate()` stub with full implementation**

Replace the `consolidate()` function in `src/triz_ai/evolution/self_evolve.py`:

```python
def consolidate(
    llm_client: LLMClient,
    store: PatentRepository,
    retention_days: int | None = None,
    min_observations: int = 3,
    source_confidence_weight: float | None = None,
) -> ConsolidationResult:
    """Consolidate search observations into matrix observations and candidates.

    Steps:
    1. Load unconsolidated observations
    2. Group by (improving_param, worsening_param)
    3. LLM validates principle assignments per group
    4. Record matrix observations (with source confidence discount)
    5. Cluster low-confidence observations for candidate discovery
    6. Mark consolidated and prune
    """
    from triz_ai.config import load_config
    from triz_ai.knowledge.parameters import get_parameter

    config = load_config()
    if retention_days is None:
        retention_days = config.evolution.retention_days
    if source_confidence_weight is None:
        source_confidence_weight = config.evolution.source_confidence_weight

    observations = store.get_unconsolidated_observations()
    if not observations:
        return ConsolidationResult()

    # Group by contradiction pair
    groups: dict[tuple[int | None, int | None], list[SearchObservation]] = {}
    for obs in observations:
        key = (obs.improving_param, obs.worsening_param)
        if key not in groups:
            groups[key] = []
        groups[key].append(obs)

    matrix_obs_added = 0
    all_low_confidence: list[SearchObservation] = []

    for (improving, worsening), group_obs in groups.items():
        # Skip non-contradiction groups for matrix recording
        if improving is None or worsening is None:
            all_low_confidence.extend(group_obs)
            continue

        # Collect principle IDs from all observations in this group
        principle_set: set[int] = set()
        for obs in group_obs:
            principle_set.update(obs.principle_ids)

        if not principle_set:
            all_low_confidence.extend(group_obs)
            continue

        # Get parameter names for the LLM prompt
        imp_param = get_parameter(improving)
        wor_param = get_parameter(worsening)
        imp_name = imp_param.name if imp_param else f"Parameter {improving}"
        wor_name = wor_param.name if wor_param else f"Parameter {worsening}"

        # LLM validates principle assignments
        try:
            validation = llm_client.validate_observations(
                observations=[
                    {"id": o.id, "title": o.title, "snippet": o.snippet}
                    for o in group_obs
                ],
                improving_param=improving,
                improving_name=imp_name,
                worsening_param=worsening,
                worsening_name=wor_name,
                principle_ids=sorted(principle_set),
            )
        except Exception:
            logger.warning(
                "Observation validation failed for (%d, %d), skipping",
                improving, worsening,
            )
            continue

        # Aggregate validated confidence per principle
        principle_scores: dict[int, list[float]] = {}
        low_conf_obs_ids: set[str] = set()

        for v in validation.validations:
            has_high_conf = False
            for vp in v.validated_principles:
                if vp.confidence >= config.evolution.review_threshold:
                    has_high_conf = True
                    if vp.principle_id not in principle_scores:
                        principle_scores[vp.principle_id] = []
                    principle_scores[vp.principle_id].append(vp.confidence)

            if not has_high_conf:
                low_conf_obs_ids.add(v.observation_id)

        # Record matrix observations for principles with enough evidence
        for principle_id, scores in principle_scores.items():
            if len(scores) >= min_observations:
                avg_conf = sum(scores) / len(scores)
                weighted_conf = avg_conf * source_confidence_weight
                # Record one observation per validated observation
                for obs in group_obs:
                    if obs.id not in low_conf_obs_ids:
                        store.insert_matrix_observation(
                            improving=improving,
                            worsening=worsening,
                            principle_id=principle_id,
                            patent_id=obs.id,
                            confidence=weighted_conf,
                        )
                        matrix_obs_added += 1

        # Collect low-confidence observations for candidate discovery
        for obs in group_obs:
            if obs.id in low_conf_obs_ids:
                all_low_confidence.append(obs)

    # Candidate principle discovery from low-confidence observations
    candidates_proposed = 0
    if len(all_low_confidence) >= min_observations:
        try:
            snippets = [
                f"{o.title}\n{o.snippet or ''}" for o in all_low_confidence
            ]
            clusters = llm_client.cluster_patents(snippets)
            for cluster_indices in clusters:
                if len(cluster_indices) < min_observations:
                    continue
                cluster_texts = [
                    snippets[i] for i in cluster_indices if i < len(snippets)
                ]
                if len(cluster_texts) < min_observations:
                    continue
                try:
                    proposal = llm_client.propose_candidate_principle(cluster_texts)
                    from triz_ai.patents.store import CandidatePrinciple

                    existing = store.get_pending_candidates()
                    next_id = len(existing) + 1
                    candidate = CandidatePrinciple(
                        id=f"C{next_id}",
                        name=proposal.name,
                        description=proposal.description,
                        evidence_patent_ids=[
                            all_low_confidence[i].id
                            for i in cluster_indices
                            if i < len(all_low_confidence)
                        ],
                        confidence=proposal.confidence,
                    )
                    store.insert_candidate_principle(candidate)
                    candidates_proposed += 1
                    logger.info(
                        "Proposed candidate principle from web observations: %s — %s",
                        candidate.id, candidate.name,
                    )
                except Exception:
                    logger.warning("Failed to propose candidate for cluster, skipping")
        except Exception:
            logger.warning("Clustering low-confidence observations failed, skipping")

    # Mark all as consolidated and prune
    store.mark_observations_consolidated([o.id for o in observations])
    pruned = store.prune_observations(retention_days=retention_days)

    result = ConsolidationResult(
        observations_processed=len(observations),
        matrix_observations_added=matrix_obs_added,
        candidate_principles_proposed=candidates_proposed,
        observations_pruned=pruned,
    )
    logger.info(
        "Consolidation complete: %d processed, %d matrix obs added, "
        "%d candidates proposed, %d pruned",
        result.observations_processed,
        result.matrix_observations_added,
        result.candidate_principles_proposed,
        result.observations_pruned,
    )
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_self_evolve.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS — no regressions

- [ ] **Step 6: Commit**

```bash
git add src/triz_ai/evolution/self_evolve.py tests/test_self_evolve.py
git commit -m "feat: implement full consolidation pipeline with LLM validation and candidate discovery"
```

---

### Task 8: CLI — `triz-ai consolidate` Command

**Files:**
- Modify: `src/triz_ai/cli.py`
- Test: Manual verification (CLI tests follow existing pattern of no automated CLI tests)

- [ ] **Step 1: Add the consolidate command**

Add to `src/triz_ai/cli.py`, near the other `@app.command()` entries:

```python
@app.command()
def consolidate(
    retention_days: int = typer.Option(
        None, help="Prune consolidated observations older than N days (default: from config)"
    ),
    model: str = typer.Option(None, help="LLM model string (overrides config)"),
    format: str = typer.Option("text", help="Output format: text, json, markdown"),
) -> None:
    """Consolidate web search observations into matrix data and candidate principles.

    Processes unconsolidated search observations collected during analyze runs.
    Validates principle assignments via LLM, records matrix observations,
    and proposes candidate principles from low-confidence clusters.
    """
    from triz_ai.evolution.self_evolve import consolidate as run_consolidate

    llm_client = _get_llm_client(model)
    store = _get_store()

    try:
        result = run_consolidate(
            llm_client,
            store,
            retention_days=retention_days,
        )
    except Exception as e:
        console.print(f"[red]Consolidation failed: {e}[/red]")
        raise typer.Exit(1) from None

    if format != "text":
        _output(result.model_dump(), format)
        return

    if result.observations_processed == 0:
        console.print("[yellow]No unconsolidated observations to process.[/yellow]")
        return

    console.print(
        Panel(
            f"[green]Observations processed:[/green] {result.observations_processed}\n"
            f"[green]Matrix observations added:[/green] {result.matrix_observations_added}\n"
            f"[green]Candidate principles proposed:[/green] {result.candidate_principles_proposed}\n"
            f"[green]Observations pruned:[/green] {result.observations_pruned}",
            title="Consolidation Results",
        )
    )

    if result.candidate_principles_proposed > 0:
        console.print(
            "\nRun [cyan]triz-ai evolve --review[/cyan] to accept or reject candidates."
        )
```

- [ ] **Step 2: Verify help text renders**

Run: `uv run triz-ai consolidate --help`
Expected: Shows usage with `--retention-days`, `--model`, `--format` options

- [ ] **Step 3: Verify command runs with no observations**

Run: `uv run triz-ai consolidate`
Expected: `No unconsolidated observations to process.`

- [ ] **Step 4: Commit**

```bash
git add src/triz_ai/cli.py
git commit -m "feat: add triz-ai consolidate CLI command for on-demand self-evolution"
```

---

### Task 9: Integration Test — End-to-End Self-Evolution Flow

**Files:**
- Test: `tests/test_self_evolve.py`

- [ ] **Step 1: Write the end-to-end integration test**

Add to `tests/test_self_evolve.py`:

```python
def test_end_to_end_collect_then_consolidate(store):
    """Full flow: collect from multiple analyses, then consolidate."""
    mock_llm = MagicMock()
    mock_llm.validate_observations.return_value = ObservationValidationBatch(
        validations=[
            ObservationValidation(
                observation_id=f"ws:{chr(97 + i) * 3}",
                validated_principles=[
                    ValidatedPrinciple(principle_id=35, confidence=0.85),
                ],
            )
            for i in range(4)
        ]
    )
    mock_llm.cluster_patents.return_value = []

    # Simulate 4 analyze calls, each producing 1 web result
    for i in range(4):
        result = AnalysisResult(
            problem=f"Problem {i}",
            method="technical_contradiction",
            improving_param={"id": 17, "name": "Temperature"},
            worsening_param={"id": 14, "name": "Strength"},
            recommended_principles=[{"id": 35, "name": "Parameter changes", "description": "..."}],
            contradiction_confidence=0.8,
            patent_examples=[
                {
                    "title": f"Web Result {i}",
                    "abstract": f"Technique {i} for thermal management",
                    "url": f"https://example.com/{i}",
                    "source": "web_search",
                },
            ],
        )
        collect_search_observations(result, store)

    # Verify 4 observations stored
    assert len(store.get_unconsolidated_observations()) == 4
    assert store.get_analyses_since_consolidation() == 4

    # Run consolidation
    consolidation_result = consolidate(mock_llm, store)

    assert consolidation_result.observations_processed == 4
    assert consolidation_result.matrix_observations_added >= 1
    assert len(store.get_unconsolidated_observations()) == 0


def test_auto_consolidation_trigger(store):
    """maybe_auto_consolidate should trigger when threshold is reached."""
    mock_llm = MagicMock()
    mock_llm.validate_observations.return_value = ObservationValidationBatch(validations=[])
    mock_llm.cluster_patents.return_value = []

    from triz_ai.evolution.self_evolve import maybe_auto_consolidate

    # Below threshold — should not trigger
    for _ in range(3):
        store.increment_analysis_count()
    result = maybe_auto_consolidate(mock_llm, store)
    assert result is None

    # Manually set counter above default threshold (25)
    conn = store._get_conn()
    conn.execute(
        "UPDATE self_evolution_meta SET analyses_since_consolidation = 25 WHERE id = 1"
    )
    conn.commit()

    # Insert at least one observation so consolidation has something to do
    obs = SearchObservation(
        id="ws:trigger",
        title="Trigger Result",
        observed_at="2026-04-13T10:00:00",
    )
    store.insert_search_observation(obs)

    result = maybe_auto_consolidate(mock_llm, store)
    assert result is not None
    assert result.observations_processed == 1
    assert store.get_analyses_since_consolidation() == 0  # counter reset
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_self_evolve.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 4: Run type check**

Run: `uvx ty check src/`
Expected: No new type errors

- [ ] **Step 5: Run pre-commit hooks**

Run: `uv run pre-commit run --all-files`
Expected: All checks pass

- [ ] **Step 6: Commit**

```bash
git add tests/test_self_evolve.py
git commit -m "test: add end-to-end integration tests for self-evolution collect-consolidate flow"
```

---

### Task 10: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add self-evolution documentation to CLAUDE.md**

Add a new section after the "### Pluggable Research Tools (Stage-Aware)" section:

```markdown
### Usage-Driven Self-Evolution

The system learns from web search results encountered during `analyze` calls. When research tools provide web results, they are captured as **search observations** in the database. These are periodically consolidated into matrix observations and candidate principles.

- **Collection**: Automatic — every `analyze` call with research tools stores web results as observations (zero latency cost, no LLM calls)
- **Consolidation triggers**: Automatic (every `consolidation_interval` analyses, default 25) or on-demand (`triz-ai consolidate`)
- **Consolidation pipeline**: LLM validates principle assignments → records `matrix_observations` (with `source_confidence_weight` discount, default 0.6) → clusters low-confidence observations for candidate principle discovery
- **Retention**: Consolidated observations are pruned after `retention_days` (default 180)
- **No patent DB required**: Self-evolution works from day one with just web search research tools
- **Config**: `evolution.consolidation_interval`, `evolution.retention_days`, `evolution.source_confidence_weight` in `~/.triz-ai/config.yaml`
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add self-evolution section to CLAUDE.md"
```
