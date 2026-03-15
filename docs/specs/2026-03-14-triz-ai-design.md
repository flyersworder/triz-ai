# triz-ai: AI-Powered TRIZ Innovation Engine

## Overview

`triz-ai` is an open-source Python CLI tool and library that combines Altshuller's TRIZ methodology with AI and real patent data to analyze technical problems, generate inventive solutions, and discover new innovation principles from modern patents.

### What makes this different

- **Evolving principles and parameters** — Existing TRIZ+AI tools (AutoTRIZ, TRIZ Agents) use the static 40 principles and 39 parameters. `triz-ai` extends parameters to 50 (adding modern domains like security, sustainability, scalability) and uses AI to discover candidate new principles and parameters from modern patents, continuing Altshuller's original work.
- **Patent-grounded** — Every suggestion is backed by real patent evidence, not just LLM generation.
- **Open-source CLI** — Existing tools are academic papers. This is usable software anyone can `pip install`.

### Target audience (MVP)

Portfolio/demo piece — optimized for demo-ability, publishable insights, and showcasing AI engineering skills. Future iterations target inventors, R&D engineers, and patent professionals.

## Architecture

### Project structure

```
triz-ai/
├── src/
│   └── triz_ai/
│       ├── __init__.py
│       ├── cli.py                  # CLI entry point (typer)
│       ├── knowledge/
│       │   ├── __init__.py
│       │   ├── principles.py       # 40 TRIZ principles as structured data
│       │   ├── contradictions.py   # Contradiction matrix (39x39 core, extensible)
│       │   ├── matrix_builder.py   # LLM-seeds missing matrix cells (params 40-50)
│       │   └── parameters.py       # 50 engineering parameters
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── analyzer.py         # Problem -> contradiction -> principles
│       │   ├── classifier.py       # Patent text -> TRIZ principle tags
│       │   ├── generator.py        # White space / idea generation
│       │   └── evaluator.py        # Idea scoring against prior art
│       ├── patents/
│       │   ├── __init__.py
│       │   ├── store.py            # Patent storage + vector search
│       │   └── ingest.py           # Patent ingestion pipeline
│       ├── evolution/
│       │   ├── __init__.py
│       │   ├── pipeline.py         # Batch classification + trend detection
│       │   └── review.py           # Human review queue for candidates
│       └── llm/
│           ├── __init__.py
│           ├── client.py           # litellm wrapper (completions + embeddings)
│           └── prompts.py          # Prompt templates with TRIZ context injection
├── data/
│   └── triz/                       # Static TRIZ knowledge (JSON)
│       ├── principles.json         # 40 principles with sub-principles
│       ├── parameters.json         # 50 engineering parameters (1-39 classic + 40-50 modern)
│       └── matrix.json             # Contradiction matrix (covers params 1-39)
├── tests/
├── pyproject.toml
└── README.md
```

### Key technology choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Package manager | uv | Fast, modern Python package management |
| CLI framework | typer | Auto-generated help, modern Python CLI |
| LLM integration | litellm | Unified interface to 100+ providers (OpenRouter, Ollama, Anthropic, OpenAI, Google, etc.) — handles completions and embeddings |
| Storage | SQLite + sqlite-vec | Zero infrastructure, portable, vector search built in |
| Data models | pydantic | JSON schema validation, structured LLM output |
| Terminal UI | rich | Tables, progress bars, formatted output |
| PDF parsing | pdfplumber | Lightweight PDF text extraction for patent documents |

## Error Handling

All external service failures (LLM providers, embedding services) raise user-friendly CLI errors with actionable guidance (e.g., "Ollama unreachable at localhost:11434 — is it running?"). litellm provides built-in exception types for common failures (rate limits, auth errors, timeouts). LLM responses are validated against pydantic schemas; malformed responses trigger a single retry with a stricter prompt, then fail with the raw response logged for debugging. No silent failures — every error surfaces to the user with context.

## TRIZ Knowledge Base

### Principles (40)

Each principle stored as structured data:

```python
class Principle(BaseModel):
    id: int              # 1-40
    name: str            # e.g., "Segmentation"
    description: str     # Core definition
    sub_principles: list[str]  # Specific techniques
    examples: list[str]  # Classic examples
    keywords: list[str]  # For text matching
```

Source data in `data/triz/principles.json`, loaded into pydantic models at runtime.

### Engineering Parameters (50)

```python
class Parameter(BaseModel):
    id: int              # 1-50
    name: str            # e.g., "Weight of moving object"
    description: str
```

Parameters 1-39 are Altshuller's original set. Parameters 40-50 are modern extensions inspired by Mann's Matrix 2010, covering domains like function efficiency (40), harmful emissions (42), security (44), safety (45), sustainability (46), scalability (49), and aesthetics (50).

### Contradiction Matrix

- Asymmetric matrix: improving param A while worsening B != improving B while worsening A
- Core matrix covers parameters 1-39 (stored as JSON, ~1455 cells at 98% fill rate)
- Each cell contains up to 4 recommended principle IDs
- Loaded as `dict[tuple[int, int], list[int]]` mapping `(improving, worsening) -> [principle_ids]`
- **Hybrid extension for params 40-50**: Missing cells (~968) can be LLM-seeded via `triz-ai matrix seed`, then refined over time by patent-observed data
- `lookup_with_observations()` merges static matrix entries with patent observations (≥3 supporting patents required), scoring by observation count with a bonus for static-matrix agreement

### Candidate Principles (evolution output)

```python
class CandidatePrinciple(BaseModel):
    id: str              # "C1", "C2", ...
    name: str
    description: str
    evidence: list[str]  # Patent IDs supporting this
    confidence: float
    status: str          # pending_review | accepted | rejected
```

Accepted candidates are promoted alongside the original 40.

### Candidate Parameters (parameter evolution output)

```python
class CandidateParameter(BaseModel):
    id: str              # "P1", "P2", ...
    name: str
    description: str
    evidence: list[str]  # Patent IDs supporting this
    confidence: float
    status: str          # pending_review | accepted | rejected
```

Discovered when patents have contradictions that don't map well to the existing 50 parameters.

## CLI Commands

### First-run workflow

Commands that query patents (`analyze`, `classify`, `discover`, `evolve`) require data in the database. On first run, seed it with: `triz-ai ingest data/patents/`. The `analyze` command works without patent data (it still does TRIZ contradiction analysis) but patent examples will be empty.

### `triz-ai analyze "problem description"`

The flagship command. Full TRIZ pipeline:

1. LLM extracts the technical contradiction (improving vs worsening parameter)
2. Maps to closest engineering parameters from the 50
3. Looks up contradiction matrix -> recommended principles
4. Searches patent store for examples of those principles applied
5. Returns: contradiction, principles, patent examples, suggested solution directions

### `triz-ai classify <source>`

Accepts a file path (`.txt`, `.pdf`), a quoted text string, or `-` for stdin. Classifies a patent through TRIZ lens:

- Which principle(s) does this patent employ?
- What contradiction does it resolve?
- Confidence score
- Stores classification in local database

### `triz-ai discover --domain "battery technology"`

White space analysis:

- Aggregates principle usage stats for patents in a domain
- Identifies underused principles
- Generates novel idea directions by applying underused principles to the domain
- Outputs a report with tables

### `triz-ai evolve`

The evolution pipeline (semi-automated):

- Batch-classifies unprocessed patents via `classify`
- Patents with classification confidence below `evolution.review_threshold` (default 0.7) are flagged as "poorly mapped"
- Poorly-mapped patents are grouped by the LLM: their abstracts are sent to the LLM in batches with a prompt asking it to identify common inventive patterns that don't fit existing principles. This is LLM-based semantic clustering, not algorithmic clustering — simpler and more interpretable for MVP.
- Minimum 3 patents must share a pattern for the LLM to propose a candidate principle
- Candidates are added to the review queue
- `triz-ai evolve --review` for interactive accept/reject

#### Parameter evolution

`triz-ai evolve --parameters` runs a parallel pipeline for discovering candidate new engineering parameters:

- Same classify → filter → cluster → propose flow as principle evolution
- Focuses on contradictions that map poorly to existing parameters (rather than principles)
- Proposes candidate parameters (IDs "P1", "P2", ...) stored in `candidate_parameters` table
- `triz-ai evolve --parameters --review` for interactive accept/reject

### `triz-ai matrix seed` / `triz-ai matrix stats`

Matrix management commands:

- `triz-ai matrix seed` — LLM-seeds missing contradiction matrix cells (params 40-50 × all params). Batches by improving parameter with a progress bar. Validates principle IDs (1-40, max 4 per cell).
- `triz-ai matrix seed --force` — Re-seeds all cells involving params 40-50, overwriting existing entries.
- `triz-ai matrix stats` — Shows fill rate, patent observation counts, and top observed parameter pairs.

Classification automatically records matrix observations: each `classify` call inserts (improving, worsening, principle_id, patent_id, confidence) into `matrix_observations`. Over time, these observations refine the LLM-seeded entries — `lookup_with_observations()` merges both sources, requiring ≥3 patent observations to influence results.

### `triz-ai ingest <source>`

Patent data ingestion:

- Supports text files, PDFs, JSON batches
- Embeds and stores in local database
- Future: bulk import from USPTO/Google Patents

### Output format

All commands support `--format` flag:
- `text` (default) — rich-formatted terminal output with colors and tables
- `json` — structured JSON to stdout for piping to other tools or future MCP/API wrapping
- `markdown` — raw markdown syntax to stdout for saving to files or embedding in docs

## Database Schema

SQLite + sqlite-vec, single file at `~/.triz-ai/patents.db`:

```sql
-- Raw patent data
CREATE TABLE patents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    claims TEXT,
    domain TEXT,
    filing_date TEXT,
    source TEXT  -- "curated" | "uspto" | "google_patents"
);

-- Vector embeddings for similarity search
CREATE VIRTUAL TABLE patent_embeddings USING vec0(
    patent_id TEXT PRIMARY KEY,
    embedding FLOAT[768]  -- nomic-embed-text dimension; changing embedding model requires re-creating the DB
);

-- TRIZ classification results
CREATE TABLE classifications (
    patent_id TEXT REFERENCES patents(id),
    principle_ids JSON,        -- [1, 14, 35]
    contradiction JSON,        -- {"improving": 9, "worsening": 1}
    confidence REAL,
    classified_at TEXT,
    PRIMARY KEY (patent_id)
);

-- Evolution pipeline output: candidate principles
CREATE TABLE candidate_principles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    evidence_patent_ids JSON,  -- ["US123", "US456"]
    confidence REAL,
    status TEXT DEFAULT 'pending_review',
    created_at TEXT
);

-- Evolution pipeline output: candidate parameters
CREATE TABLE candidate_parameters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    evidence_patent_ids JSON,
    confidence REAL,
    status TEXT DEFAULT 'pending_review',
    created_at TEXT
);

-- Patent-observed matrix refinement: principle-contradiction associations from classified patents
CREATE TABLE matrix_observations (
    improving_param INTEGER NOT NULL,
    worsening_param INTEGER NOT NULL,
    principle_id INTEGER NOT NULL,
    patent_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    observed_at TEXT,
    PRIMARY KEY (improving_param, worsening_param, principle_id, patent_id)
);
```

## LLM Integration

### litellm client

`llm/client.py` wraps litellm to provide a unified interface for both completions and embeddings. litellm supports 100+ providers through a single `completion()` and `embedding()` API — switching providers is just changing the model string.

Six core LLM interactions:

1. **extract_contradiction**(problem_text) -> `{improving_param, worsening_param, reasoning}`
2. **classify_patent**(patent_text) -> `{principle_ids, contradiction, confidence, reasoning}`
3. **generate_ideas**(domain, underused_principles, existing_patents) -> `{ideas: [...]}`
4. **propose_candidate_principle**(patent_cluster) -> `{name, description, how_it_differs, confidence}`
5. **propose_candidate_parameter**(patent_cluster) -> `{name, description, how_it_differs, confidence}`
6. **seed_matrix_row**(improving, worsening_params) -> `{entries: [{improving, worsening, principles}]}` — fills missing contradiction matrix cells

Embeddings:

- `litellm.embedding(model, input)` handles all embedding calls
- Works with Ollama (`ollama/nomic-embed-text`), OpenAI, Cohere, etc.
- No separate embedding client needed

### Prompt strategy

- System prompt includes relevant TRIZ context (principles, parameters, matrix subset)
- JSON output schemas for reliable parsing
- Few-shot examples for consistency
- TRIZ knowledge injected into prompt rather than relying on LLM training data — ensures accuracy and includes evolved principles
- Token budget: only inject the relevant matrix row/column, relevant principles (not all 40), and parameter descriptions for the identified contradiction. Keeps system prompts under ~2K tokens.

### Model selection

- Default: `openrouter/google/gemini-2.5-flash` (capable, affordable, via OpenRouter)
- Configurable via config file or `--model` flag
- Any litellm-supported model string works without code changes (e.g., `anthropic/claude-sonnet-4-6`, `ollama/llama3`, `openai/gpt-4o`)

## Configuration

`~/.triz-ai/config.yaml`:

```yaml
llm:
  default_model: openrouter/google/gemini-2.5-flash  # any litellm model string

embeddings:
  model: ollama/nomic-embed-text  # any litellm embedding model string

database:
  path: ~/.triz-ai/patents.db

evolution:
  auto_classify: true
  review_threshold: 0.7
```

API keys are set via environment variables following litellm conventions (e.g., `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). litellm reads these automatically — no key management code needed.

## Schema Versioning

MVP: no migrations. Schema changes require deleting and re-creating `patents.db` (`triz-ai init --force`). Post-MVP: add alembic or simple version-table-based migrations.

## Initial Dataset

MVP starts with a curated set of EV/battery patents (50-100). This is enough to demonstrate the full pipeline end-to-end. Expansion to USPTO/Google Patents bulk data is a future step via the `ingest` command.

## Future Extensions (Post-MVP)

- **MCP server** — wrap CLI as MCP tools for Claude Desktop integration
- **REST API** — FastAPI wrapper for web access
- **Web UI** — simple frontend for public demos
- **Bulk patent sources** — Google Patents via BigQuery, USPTO XML dumps
- **Scheduled pipeline** — cronjob on VPS for continuous patent ingestion and classification
- **Publishable reports** — auto-generated "TRIZ Evolution Reports" as blog content
