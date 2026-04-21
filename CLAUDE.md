# CLAUDE.md

## Build & Development

```bash
uv sync                          # Install dependencies
uv run triz-ai <command>         # Run CLI
uv run pytest                    # Run all tests
uv run pytest tests/test_foo.py  # Run single test file
uv run pytest -k "test_name"    # Run single test by name
uvx ty check src/                # Type check
uv run pre-commit run --all-files  # Run all pre-commit hooks
```

## Architecture

`src/triz_ai/` modules: `cli.py` (Typer CLI) → `engine/router.py` (problem classifier + IFR + RCA + dispatch) → `engine/` (analyzer, physical, su_field, function_analysis, trimming, trends, classifier, generator, evaluator) → `llm/client.py` (openai SDK + optional litellm) → `patents/` (`PatentRepository` protocol + SQLite-backed `PatentStore` default, pluggable `VectorStore` protocol, ingestion, matrix observations) → `knowledge/` (TRIZ data from `src/triz_ai/data/*.json`, `matrix_builder.py` for LLM-seeding) → `evolution/` (candidate principle and parameter discovery).

### Pluggable Patent Repository

`patents/repository.py` defines a `PatentRepository` protocol (28 methods) covering patents, classifications, candidate principles/parameters, matrix observations, and self-evolution (search observations + meta tracking). `PatentStore` (SQLite-backed) is the default implementation. All engine/evolution consumers type-hint `PatentRepository`, not `PatentStore` — alternative backends (Postgres, DynamoDB, etc.) implement this protocol for full database portability. `cli.py` remains the concrete factory, creating `PatentStore()`.

### Pluggable Vector Database

`patents/vector.py` defines a `VectorStore` protocol with 4 methods (`init`, `insert`, `search`, `close`). Default `SqliteVecStore` wraps sqlite-vec. `PatentStore` accepts an optional `vector_store` parameter — if not provided, creates `SqliteVecStore` over the same db file; each class owns its own thread-local `sqlite3.Connection` so both are safe from multi-threaded callers (Flask/Gunicorn, `ThreadPoolExecutor`). Alternative backends (Chroma, Qdrant, pgvector) can be plugged via `PatentStore(vector_store=my_store)`. Hybrid scoring (TRIZ domain logic) stays in `PatentStore`.

### Multi-Tool Routing

`analyze` auto-classifies problems and routes to the best TRIZ pipeline:
- `technical_contradiction` → `analyzer.py` (improve X without worsening Y)
- `physical_contradiction` → `physical.py` (part must be A AND B)
- `su_field` → `su_field.py` (detection/measurement/interaction problems)
- `function_analysis` → `function_analysis.py` (harmful component interactions)
- `trimming` → `trimming.py` (simplification/cost reduction)
- `trends` → `trends.py` (technology evolution + system operator)

IFR is always formulated first. If classifier confidence < 0.4, RCA reformulates before re-routing. `--method` flag bypasses classifier. `--router-model` overrides classification model.

### Deep ARIZ-85C Mode (`--deep`)

`analyze --deep` bypasses the router entirely and runs a 3-pass ARIZ-85C orchestrator (`engine/ariz.py`):
- **Pass 1**: Single LLM call reformulates the problem deeply — identifies both TCs (intensified), physical contradiction (macro+micro), IFR, resource inventory, and recommends 2-4 tools
- **Pass 2**: Runs selected pipelines in parallel via `ThreadPoolExecutor` (IO-bound, GIL not an issue)
- **Pass 3**: Verifies each candidate against IFR, scores ideality, synthesizes best elements
- **Escape hatch**: If no candidate satisfies IFR, swaps TC1↔TC2 and re-runs Passes 2-3 once
- `--deep` and `--method` are mutually exclusive
- `deep_model` and `reasoning_effort` are configurable in `~/.triz-ai/config.yaml` under `llm`; CLI flags `--deep-model` and `--reasoning-effort` override config
- Pass 2 pipelines always use `default_model` (via `--model`); Passes 1 & 3 use `deep_model` (falls back to `default_model`)
- `reasoning_effort` accepts `low|medium|high`; litellm translates across providers (Anthropic, OpenAI o-series, DeepSeek R1, etc.)

### Pluggable Research Tools (Stage-Aware)

`ResearchTool` dataclass (`tools.py`) lets developers pass research tools that run at specific pipeline stages. `fn` signature: `(query: str, context: dict) -> list[dict]`.

- **Three stages**: `"context"` (before LLM extraction), `"search"` (during patent search), `"enrichment"` (after solution generation). Default: `["search"]`.
- **Context stage**: Runs once in `route()` / `orchestrate_deep()` before dispatch. Returns `[{"content": "..."}]`, prepended to `problem_text`.
- **Search stage**: Runs in `search_patents()`, filtered by `"search"` stage. Results deduplicated by title, tagged with `source`. Returns `[{"title": "...", "abstract": "..."}]`.
- **Enrichment stage**: Runs after `generate_solution_directions()` in each pipeline. Stored in `AnalysisResult.enrichment`. Returns `[{"title": "...", "content": "..."}]`.
- **`run_stage_tools()`**: Filters tools by stage, passes context dict; exported from `triz_ai` package.
- **Deep mode**: LLM selects which research tools to use via `recommended_research_tools`; tool descriptions include stages.
- Tool failures are logged and skipped — they never block analysis
- No CLI changes; research tools are passed programmatically via `route(research_tools=[...])` or `orchestrate_deep(research_tools=[...])`

### Usage-Driven Self-Evolution

The system learns from web search results encountered during `analyze` calls. When research tools provide web results, they are captured as **search observations** in the database. These are periodically consolidated into matrix observations and candidate principles.

- **Collection**: Automatic — every `analyze` call with research tools stores web results as observations (zero latency cost, no LLM calls)
- **Consolidation triggers**: Automatic (every `consolidation_interval` analyses, default 25) or on-demand (`triz-ai consolidate`)
- **Consolidation pipeline**: LLM validates principle assignments → records `matrix_observations` (with `source_confidence_weight` discount, default 0.6) → clusters low-confidence observations for candidate principle discovery
- **Retention**: Consolidated observations are pruned after `retention_days` (default 180)
- **No patent DB required**: Self-evolution works from day one with just web search research tools
- **Config**: `evolution.consolidation_interval`, `evolution.retention_days`, `evolution.source_confidence_weight` in `~/.triz-ai/config.yaml`

## Key Constraints

- **6 TRIZ analysis methods** — technical contradiction, physical contradiction, Su-Field, function analysis, trimming, trends. Router auto-classifies; `--method` forces one.
- **50 engineering parameters** — IDs 1-39 are Altshuller's originals, 40-50 are modern extensions (Mann's Matrix 2010). The static contradiction matrix covers 1-39; cells for 40-50 can be LLM-seeded via `triz-ai matrix seed` and refined by patent observations over time. `lookup_with_observations()` merges both sources.
- **Contradiction matrix is asymmetric** — improving A worsening B ≠ improving B worsening A
- **Embedding dimension is 768** — changing embedding model requires `triz-ai init --force`
- **LLM responses validated via pydantic** — malformed → 1 retry with stricter prompt, then fail
- **Auto-init** — `analyze` and other commands work without running `init` first; `init` is only needed with `--force` to reset the database
- **Hybrid patent search** — `analyze` (technical contradiction) uses `search_patents_hybrid()` which combines vector similarity with principle overlap bonus (0.3/principle, cap 0.6) and contradiction match bonus (0.4 exact, 0.2 partial). Other methods use vector-only search.
- **No DB migrations** — schema changes require `triz-ai init --force`
- **Token budget** — only inject relevant matrix rows/principles into prompts, not full dataset. Parameters include descriptions for disambiguation.
- **TRIZ knowledge data** — `separation_principles.json` (4 categories), `standard_solutions.json` (76 solutions, 5 classes), `evolution_trends.json` (8 trends with stages)

## Release Workflow

- Version source: `pyproject.toml` and `CHANGELOG.md`. Feature additions → minor bump (0.15.0 → 0.16.0); bugfixes → patch bump. `uv.lock` regenerates on any version change — include it in the release commit.
- `gh release create vX.Y.Z --target main --title "..." --notes "..."` tags main and triggers `.github/workflows/ci-and-publish.yml` (test → publish). The publish job runs `uv build` + `pypa/gh-action-pypi-publish`.
- **PyPI uploads are irreversible** — a filename, once uploaded and later deleted, cannot be re-uploaded. Only create a release after the PR is merged, CI is green on main, and you've confirmed the user wants to publish.

## Models

- Default LLM: `openrouter/nvidia/nemotron-3-super-120b-a12b:free`
- Default classify model: `openrouter/nvidia/nemotron-3-nano-30b-a3b:free` (smaller model used for patent classification during ingest)
- Default embeddings: `openrouter/nvidia/llama-nemotron-embed-vl-1b-v2:free`
- Alternative classify model: `openrouter/google/gemini-3.1-flash-lite-preview` — not free but extremely cheap ($0.25/$1.50 per M tokens), works within OpenRouter's default free testing allowance. Use via `--classify-model` flag or `llm.classify_model` in config. `classify_patent()` sets `max_tokens=1024` to avoid reserving the full 65K output window against credits.

## References

- Design spec: `docs/specs/2026-03-14-triz-ai-design.md`
- Config: `~/.triz-ai/config.yaml` (default), overridable via `--config` CLI flag or `TRIZ_AI_CONFIG` env var. Values support `${VAR}` and `${VAR:-default}` env var interpolation; `$$` escapes a literal `$`. Unset/empty `${VAR}` (no default) fails at startup with a field-path error (e.g. `llm.api_key: environment variable LITELLM_MASTER_KEY is not set`).
- API keys: `.env` file (loaded via python-dotenv) or env vars per litellm conventions (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, etc.)
- SSL: For corporate proxies with internal CA certs, set `llm.ssl_verify: false` in config. This creates a custom OpenAI client with `httpx.Client(verify=False)` passed to litellm via `client=`.
- LLM backend: `litellm` is an optional dependency (`pip install triz-ai[litellm]`). Without it, the `openai` SDK is used as fallback and requires `api_base` in config pointing to an OpenAI-compatible endpoint (e.g. a self-hosted litellm gateway). With litellm installed, any provider works directly.
