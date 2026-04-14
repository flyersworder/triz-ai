# Usage-Driven Self-Evolution Design

**Date**: 2026-04-13
**Status**: Draft

## Overview

The self-evolution system enables triz-ai to learn from every `analyze` call without user intervention or a patent database. When research tools (web search, etc.) provide results during analysis, those results are captured as **search observations** — lightweight records associating each web result with the analysis context (contradiction pair, recommended principles, confidence). Periodically, a **consolidation step** clusters accumulated observations and distills them into matrix observations and candidate principles/parameters, improving future analyses.

### Goals

- **Zero-effort learning**: the system improves with use, no explicit `evolve` command needed
- **No patent DB required**: works from day one with just web search research tools
- **No user-facing changes**: analysis output is identical; learning is invisible
- **No latency impact**: collection is pure storage; LLM calls only happen during consolidation
- **Backward compatible**: all changes are additive to the `PatentRepository` protocol

### Non-Goals

- Built-in web search — the system collects from whatever research tools the user passes
- Replacing the patent-based evolution pipeline — that continues to work independently
- Changing analysis output format — `AnalysisResult` is unchanged

## Architecture

### Two-Phase Design: Collect then Consolidate

```
Phase 1: COLLECT (every analyze call)
─────────────────────────────────────
analyze("problem", research_tools=[web_search])
  ├─ normal analysis pipeline runs
  ├─ web search results flow through as patent_examples
  └─ collect_search_observations(result, store)
       → search_observations table
  └─ caller increments analysis counter if observations were collected

Phase 2: CONSOLIDATE (periodic or on-demand)
─────────────────────────────────────────────
Automatic: every N analyses (default 25)
On-demand: triz-ai consolidate

  ├─ load unconsolidated search_observations
  ├─ LLM validates principle assignments per observation
  ├─ record matrix observations (with source confidence discount)
  ├─ cluster low-confidence observations → propose candidate principles
  ├─ mark observations as consolidated
  └─ prune observations older than retention period
```

### Data Flow

```
User: triz-ai analyze "problem..." (with research tools)
  │
  ▼
┌────────────────────────────────┐
│  Normal analysis pipeline      │
│  (route / orchestrate_deep)    │
│  → produces AnalysisResult     │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│  collect_search_observations() │
│  • Filter patent_examples with │
│    source field (= web results)│
│  • Store with analysis context │
│  • Increment analysis counter  │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│  Auto-consolidation check      │
│  analyses_since >= threshold?  │
│  → YES: run consolidate()      │
│  → NO:  return result          │
└────────────────────────────────┘
```

## Data Model

### New Table: `search_observations`

Stores web search results from analysis runs with their analysis context.

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
```

**Field details:**

| Field | Description |
|-------|-------------|
| `id` | `"ws:<sha256_hex[:16]>"` of `title + snippet` — deduplicates identical results across analyses |
| `title` | Web result title |
| `snippet` | Abstract or snippet text from the search result |
| `url` | Source URL |
| `source_tool` | Name of the research tool that produced this result (e.g. `"web_search"`) |
| `problem_text` | The problem the user was analyzing when this result appeared |
| `analysis_method` | TRIZ method used (e.g. `"technical_contradiction"`, `"physical_contradiction"`) |
| `improving_param` | Improving parameter ID from the analysis context (NULL for non-contradiction methods) |
| `worsening_param` | Worsening parameter ID (NULL for non-contradiction methods) |
| `principle_ids` | JSON array of principle IDs the analysis recommended |
| `analysis_confidence` | Confidence score from the analysis |
| `consolidated` | 0 = pending consolidation, 1 = processed |
| `observed_at` | ISO timestamp of when the observation was recorded |
| `consolidated_at` | ISO timestamp of when consolidation processed this observation |

**Deduplication**: If the same web result appears in multiple analyses, the existing row is kept (INSERT OR IGNORE). The first analysis context is preserved — this is acceptable because the same result appearing across different problems is itself a signal of general relevance, not something we need to track per-problem.

### New Table: `self_evolution_meta`

Single-row tracking table for auto-consolidation trigger.

```sql
CREATE TABLE IF NOT EXISTS self_evolution_meta (
    id INTEGER PRIMARY KEY DEFAULT 1,
    analyses_since_consolidation INTEGER DEFAULT 0,
    last_consolidated_at TEXT
);
```

### Existing Tables: No Changes

- `matrix_observations` — consolidation writes here (same as patent classification does today)
- `candidate_principles` — consolidation writes here (same as evolution pipeline)
- `candidate_parameters` — consolidation writes here (same as evolution pipeline)

## Data Model: SearchObservation

New Pydantic model in `evolution/self_evolve.py`:

```python
class SearchObservation(BaseModel):
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
```

## Phase 1: Collection

### Hook Point

Collection runs at the end of `route()` and `orchestrate_deep()`, after the `AnalysisResult` is built but before returning it to the caller. This ensures:

- All analysis context is available (method, params, principles, confidence)
- All web search results are in `patent_examples` (tagged with `source` field)
- No additional LLM calls — collection is pure storage
- No latency impact on the user-facing analysis

### Collection Logic

```python
def collect_search_observations(
    result: AnalysisResult,
    store: PatentRepository,
) -> int:
    """Store web search results from analysis as search observations.

    Filters patent_examples to those with a 'source' field (web results),
    builds SearchObservation from each, and stores in the DB.

    Returns number of observations stored.
    """
    count = 0
    for example in result.patent_examples:
        source = example.get("source")
        if not source:
            # Local DB results have no source field — skip
            continue

        title = example.get("title", "")
        snippet = example.get("abstract", "")
        if not title:
            continue

        obs_id = _make_observation_id(title, snippet)
        obs = SearchObservation(
            id=obs_id,
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
        logger.debug("Collected %d search observations", count)

    return count


def _make_observation_id(title: str, snippet: str) -> str:
    """Generate a deterministic ID for deduplication."""
    import hashlib
    content = f"{title}|{snippet or ''}"
    hash_hex = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"ws:{hash_hex}"
```

### Integration in router.py

```python
def route(problem_text, llm_client, store=None, method=None, research_tools=None):
    # ... existing analysis logic ...
    result = pipeline(problem_text, ifr, llm_client, store, research_tools=research_tools)

    # Self-evolution: collect web search observations
    if store is not None and research_tools:
        from triz_ai.evolution.self_evolve import (
            collect_search_observations,
            maybe_auto_consolidate,
        )
        try:
            collected = collect_search_observations(result, store)
            if collected > 0:
                store.increment_analysis_count()
            maybe_auto_consolidate(llm_client, store)
        except Exception:
            logger.warning("Self-evolution collection failed, continuing", exc_info=True)

    return result
```

Same pattern in `orchestrate_deep()`.

### Failure Handling

Collection failures are logged and silently skipped — they never block analysis. This mirrors how research tool failures are handled today.

## Phase 2: Consolidation

### Consolidation Pipeline

```python
def consolidate(
    llm_client: LLMClient,
    store: PatentRepository,
    retention_days: int = 180,
    min_observations: int = 3,
    source_confidence_weight: float = 0.6,
) -> ConsolidationResult:
```

**Step 1 — Load unconsolidated observations**

```python
observations = store.get_unconsolidated_observations()
```

Group by `(improving_param, worsening_param)` pairs. Observations from non-contradiction methods (improving/worsening are NULL) are grouped separately for candidate discovery only.

**Step 2 — LLM validates principle assignments**

For each group of observations sharing a contradiction pair, ask the LLM to validate:

> "Given these web search results about [titles/snippets], do they support TRIZ principles [X, Y, Z] for the contradiction: improving parameter [A] / worsening parameter [B]? For each result, rate confidence (0-1) per principle."

This produces a validated confidence score per `(observation, principle)` pair. The LLM call batches all observations for a given contradiction pair into a single request to minimize API calls.

**Step 3 — Record matrix observations**

For each `(improving, worsening, principle_id)` triple that appears across enough validated observations (>= `min_observations`, default 3):

```python
store.insert_matrix_observation(
    improving=improving,
    worsening=worsening,
    principle_id=principle_id,
    patent_id=obs.id,  # "ws:..." synthetic ID
    confidence=avg_validated_confidence * source_confidence_weight,
)
```

The `source_confidence_weight` (default 0.6) discounts web evidence relative to patent evidence. This ensures patents remain the primary signal when both are present, while web observations still contribute to cells with no patent data.

**Step 4 — Candidate principle discovery**

Observations that the LLM couldn't confidently map to existing principles (validated confidence < `review_threshold`) are clustered using `llm_client.cluster_patents()` (reusing the existing LLM method — it works on any text, not just patent abstracts). Clusters meeting the minimum size threshold are fed to `llm_client.propose_candidate_principle()`.

This reuses the existing evolution pipeline's clustering and proposal logic entirely — the only difference is the input source (web snippets vs patent abstracts).

**Step 5 — Mark and prune**

```python
store.mark_observations_consolidated(observation_ids)
pruned = store.prune_observations(retention_days=retention_days)
```

Mark all processed observations as `consolidated = 1` with `consolidated_at` timestamp. Delete observations where `consolidated_at` is older than `retention_days`.

### ConsolidationResult

```python
class ConsolidationResult(BaseModel):
    observations_processed: int
    matrix_observations_added: int
    candidate_principles_proposed: int
    observations_pruned: int
```

## Triggers

### Automatic Trigger

After each `analyze` call that collects observations, check whether auto-consolidation should run:

```python
def maybe_auto_consolidate(
    llm_client: LLMClient,
    store: PatentRepository,
) -> ConsolidationResult | None:
    from triz_ai.config import load_config
    config = load_config()

    count = store.get_analyses_since_consolidation()
    if count < config.evolution.consolidation_interval:
        return None

    result = consolidate(
        llm_client, store,
        retention_days=config.evolution.retention_days,
        source_confidence_weight=config.evolution.source_confidence_weight,
    )
    store.reset_analysis_count()
    return result
```

The auto-consolidation runs inline at the end of `analyze`. Since it only triggers every N analyses (default 25), the occasional latency hit is acceptable. The LLM calls during consolidation use the default model.

### On-Demand Trigger

New CLI command:

```bash
triz-ai consolidate
  --retention-days 180     # override retention period
  --model <name>           # LLM model for validation
  --format text|json|markdown
```

This runs the full consolidation pipeline immediately and displays the result.

## Configuration

### New Fields on `EvolutionConfig`

```python
class EvolutionConfig(BaseModel):
    auto_classify: bool = True
    review_threshold: float = 0.7
    consolidation_interval: int = 25         # auto-consolidate every N analyses
    retention_days: int = 180                # prune consolidated observations after N days
    source_confidence_weight: float = 0.6    # web results confidence discount vs patents
```

Configurable via `~/.triz-ai/config.yaml`:

```yaml
evolution:
  consolidation_interval: 25
  retention_days: 180
  source_confidence_weight: 0.6
```

## Protocol Changes

### New Methods on `PatentRepository`

```python
@runtime_checkable
class PatentRepository(Protocol):
    # ... existing methods ...

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

The `SearchObservation` type is imported from `triz_ai.evolution.self_evolve` via `TYPE_CHECKING` in `repository.py` (same pattern as the existing `CandidatePrinciple` import from `store.py`).

All methods are additive. Existing `PatentRepository` implementations that don't support self-evolution will fail at runtime only if self-evolution code is triggered — which requires research tools to be passed. This is acceptable: library consumers who implement a custom backend and don't use research tools will never hit these methods.

## File Changes

| File | Status | Purpose |
|------|--------|---------|
| `src/triz_ai/evolution/self_evolve.py` | **New** | `SearchObservation` model, `collect_search_observations()`, `consolidate()`, `maybe_auto_consolidate()`, `ConsolidationResult` |
| `src/triz_ai/engine/router.py` | Modified | Call `collect_search_observations()` + `maybe_auto_consolidate()` after analysis |
| `src/triz_ai/engine/ariz.py` | Modified | Same for deep mode |
| `src/triz_ai/patents/repository.py` | Modified | Add 7 new protocol methods |
| `src/triz_ai/patents/store.py` | Modified | Implement new methods, add 2 new tables to schema SQL |
| `src/triz_ai/config.py` | Modified | Add 3 new fields to `EvolutionConfig` |
| `src/triz_ai/cli.py` | Modified | Add `triz-ai consolidate` command |
| `src/triz_ai/llm/client.py` | Modified | Add `validate_observations()` LLM method for consolidation Step 2 |
| `src/triz_ai/llm/prompts.py` | Modified | Add prompt template for observation validation |

## Edge Cases

### No research tools passed

If `analyze` is called without research tools (e.g., from CLI today), `result.patent_examples` will have no `source`-tagged entries. `collect_search_observations()` stores nothing, counter is not incremented. Self-evolution is inert.

### No store available

If `store is None` (user hasn't run `init`), collection is skipped. The guard `if store is not None and research_tools:` in `route()` handles this.

### Duplicate web results across analyses

The `INSERT OR IGNORE` semantics on the `id` primary key (content-based hash) means the same web result appearing in multiple analyses is stored only once. The first analysis context is preserved.

### Non-contradiction methods

Methods like su-field, trimming, and trends don't produce `improving_param` / `worsening_param`. Their observations are stored with NULL params. During consolidation, these are excluded from matrix observation recording (Step 3) but included in candidate principle discovery (Step 4) — they can still reveal novel principles even without a contradiction pair.

### Schema migration

The new tables use `CREATE TABLE IF NOT EXISTS`, so they're created on first access. No explicit migration needed. Existing databases get the new tables automatically.

### Consolidation with zero observations

`consolidate()` returns immediately with a zero-count result if there are no unconsolidated observations. No LLM calls are made.

## Testing Strategy

- **Unit tests for collection**: mock `AnalysisResult` with mixed source/no-source patent_examples, verify correct filtering and storage
- **Unit tests for consolidation**: mock stored observations, verify matrix observation output and candidate proposals
- **Unit tests for auto-trigger**: verify counter increment/reset logic and threshold check
- **Integration test**: end-to-end flow from `route()` with a mock research tool through collection and consolidation
- **Edge case tests**: no research tools, no store, duplicate observations, non-contradiction methods
