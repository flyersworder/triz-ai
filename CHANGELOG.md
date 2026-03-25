# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.14.0] - 2026-03-25

### Changed

- **`litellm` is now an optional dependency**: Install `pip install triz-ai[litellm]` for direct access to any LLM provider (OpenRouter, Anthropic, Google, Ollama, etc.). The bare `pip install triz-ai` uses the lightweight `openai` SDK and requires `api_base` pointing to an OpenAI-compatible endpoint (e.g. a self-hosted litellm gateway)
- **`openai` SDK added as core dependency**: Replaces litellm as the default LLM transport for users with an OpenAI-compatible gateway. Lightweight (~2MB vs litellm's ~50MB+)
- **Dual LLM backend in `client.py`**: `HAS_LITELLM` flag selects between litellm (any provider) and openai SDK (OpenAI-compatible endpoints). All `LLMClient` public methods unchanged — no API changes for consumers
- **Cached OpenAI clients**: OpenAI SDK clients are created once and reused across calls (also applies to litellm's SSL-disabled path)
- **Separate `api_base` validation for embeddings**: `_require_api_base(for_embeddings=True)` checks `embeddings.api_base` independently from `llm.api_base`, with a specific error message pointing to the correct config key
- **Version**: Bumped to 0.14.0

## [0.13.0] - 2026-03-23

### Added

- **`--config` CLI flag**: Global option to specify a custom config YAML path — `triz-ai --config /path/to/config.yaml analyze "..."`
- **`TRIZ_AI_CONFIG` env var**: Environment variable override for config path, useful for Docker, CI/CD, and multi-project setups
- **Config path precedence**: `--config` flag → `TRIZ_AI_CONFIG` env var → `~/.triz-ai/config.yaml` (default)
- **`ssl_verify` config option**: `llm.ssl_verify: false` disables TLS certificate verification for corporate proxies with internal CA certs. Creates a custom `openai.OpenAI` client with `httpx.Client(verify=False)` passed to litellm via `client=`
- **`load_config(config_path=)` parameter**: Programmatic API now accepts an explicit config path, complementing the env var and CLI flag
- **`set_config_path()` function**: Module-level setter for config path override, used internally by the CLI callback

### Changed

- **Version**: Bumped to 0.13.0

## [0.12.1] - 2026-03-19

### Improved

- **Prompt engineering overhaul**: Optimized 7 of 18 system prompts for better output quality across different LLMs
  - **Few-shot examples** added to `extract_contradiction`, `classify_patent`, and `classify_problem` prompts — grounds model output with concrete correct examples
  - **Structured "Analysis steps"** added to 5 prompts — lightweight chain-of-thought guidance before JSON output
  - **Confidence calibration** defined in 3 prompts — confidence ranges now aligned with downstream thresholds (e.g., 0.4 RCA trigger in router)
  - **Negative examples / common mistakes** added to 4 prompts — prevents TC/PC confusion, same-parameter-for-both errors, domain-vs-principle conflation
  - **Su-Field field type taxonomy** — 8 TRIZ field types (Mechanical, Thermal, Chemical, Electromagnetic, Optical, Gravitational, Nuclear, Biological) with sub-examples constrain LLM output to proper TRIZ terminology
  - **`classify_patent` prompt fixed** — was missing the engineering parameter list; LLM had to guess parameter IDs without reference
  - **Parameter descriptions** now included in `_parameters_list()` for better disambiguation across all parameter-referencing prompts (7 prompts)

### Changed

- **`_parameters_list()` output format**: Changed from `"ID. Name"` to `"ID. Name — Description"` for improved parameter disambiguation at modest token cost
- **Version**: Bumped to 0.12.1

## [0.12.0] - 2026-03-18

### Added

- **`PatentRepository` protocol**: New 18-method protocol (`patents/repository.py`) defines the full patent storage interface — patents, classifications, candidate principles/parameters, and matrix observations. Any class implementing this protocol can replace the default SQLite backend.
- **Pluggable database backend**: All engine, evolution, and ingestion consumers now type-hint `PatentRepository` instead of `PatentStore`. Alternative backends (Postgres, DynamoDB, etc.) can be plugged via `route(store=my_repo)` or `orchestrate_deep(store=my_repo)`.
- **`database.backend` config field**: `DatabaseConfig` gains `backend: str = "sqlite"` to record the chosen backend; no factory function yet — programmatic usage is the primary plug-in path.
- **Protocol conformance tests**: `test_repository.py` with 6 tests — `isinstance` check, method existence, mock delegation, negative conformance, and static type-checker assertion.

### Changed

- **Type hints across 15 files**: `PatentStore` type annotations replaced with `PatentRepository` in `router.py`, `analyzer.py`, `classifier.py`, `generator.py`, `evaluator.py`, `physical.py`, `su_field.py`, `function_analysis.py`, `trimming.py`, `trends.py`, `ingest.py`, `pipeline.py`, `review.py`, `contradictions.py`, and `ariz.py`
- **`contradictions.py`**: `store: object | None` upgraded to `store: PatentRepository | None`; removed stale `type: ignore[attr-defined]` suppression
- **`ariz.py`**: Added type annotations to `orchestrate_deep()` and `_run_tools()` (`store: PatentRepository | None`, `llm_client: LLMClient`)
- **`PatentRepository` exported** from `triz_ai.patents` package
- **Version**: Bumped to 0.12.0

### Note

- **Zero breaking changes**: `PatentStore` still works everywhere — it satisfies `PatentRepository` structurally. `cli.py` still creates `PatentStore()` concretely. Existing code constructing `PatentStore()` is unaffected.

## [0.11.1] - 2026-03-15

### Fixed

- **Context tools now run before IFR** in `route()` — previously IFR was formulated from un-enriched problem text while context tools ran after; now consistent with `orchestrate_deep()` where context tools already ran first
- **`VALID_STAGES` validation enforced** — `ResearchTool.__post_init__` now validates that all `stages` values are in `{"context", "search", "enrichment"}`; invalid stages raise `ValueError` immediately instead of silently never running
- **Stale docstrings updated** — `route()`, `orchestrate_deep()`, `search_patents()`, `AnalysisResult`, `run_stage_tools()`, and all 6 pipeline functions now document context/enrichment steps

## [0.11.0] - 2026-03-15

### Added

- **Stage-aware research tools**: `ResearchTool` gains a `stages` field (default `["search"]`) so tools can declare participation in `"context"` (before LLM extraction), `"search"` (during patent search), and/or `"enrichment"` (after solution generation)
- **Context stage**: Tools registered for `"context"` run once in `route()` / `orchestrate_deep()` before dispatch; their output is prepended to `problem_text` for all downstream LLM calls
- **Enrichment stage**: Tools registered for `"enrichment"` run after solution directions are generated in each pipeline; results stored in new `AnalysisResult.enrichment` field
- **`run_stage_tools()` helper**: Filters tools by stage, passes a context dict, and handles failures gracefully; exported from top-level `triz_ai` package
- **`run_enrichment_tools()` helper**: Convenience wrapper in `analyzer.py` that calls `run_stage_tools` with `"enrichment"` stage and solution directions as context

### Changed

- **Breaking: `ResearchTool.fn` signature** changed from `(str) -> list[dict]` to `(str, dict) -> list[dict]` — the second argument is a context dict containing at minimum `{"stage": str}` plus stage-specific data (e.g., `principle_ids` for search, `solution_directions` for enrichment)
- **Deep mode tool descriptions** now include `stages` field so the LLM knows what each tool can do at each stage
- **`search_patents()`** now filters research tools by `"search"` stage and passes a context dict with `principle_ids`, `improving_param`, `worsening_param`
- **Version**: Bumped to 0.11.0

## [0.10.0] - 2026-03-15

### Added

- **Pluggable research tools interface**: New `ResearchTool` dataclass lets developers pass additional research tools (web search, BigQuery, Arxiv, etc.) to supplement the built-in patent DB search
- Research tools thread through `route()` and `orchestrate_deep()` into all 6 pipelines via `search_patents()`
- In normal mode, all research tools run automatically; in deep mode, the LLM selects which tools to use based on their descriptions
- Results are deduplicated by title and tagged with a `source` field; tool failures are logged and skipped
- `ResearchTool` exported from top-level `triz_ai` package for easy access

### Changed

- **Version**: Bumped to 0.10.0

## [0.9.0] - 2026-03-15

### Added

- **Deep ARIZ-85C analysis (`--deep`)**: Full 3-pass orchestrator that reformulates problems deeply, runs 2-4 TRIZ tools in parallel, and verifies solutions against the Ideal Final Result
- **Pass 1 — Deep reformulation**: Single rich LLM call identifies both technical contradictions (intensified), physical contradiction (macro + micro), IFR, resource inventory, and recommends tools
- **Pass 2 — Multi-tool research**: 2-4 existing pipelines run in parallel via ThreadPoolExecutor; each receives the reformulated problem and IFR from Pass 1
- **Pass 3 — Verify + synthesize**: Checks each candidate against IFR, scores ideality, synthesizes best elements, identifies supersystem changes
- **Escape hatch**: If no candidate satisfies IFR, automatically swaps TC1/TC2 and re-runs Passes 2-3 (maximum 1 retry)
- **ARIZ Pydantic models**: `StructuredProblemModel`, `SolutionVerification`, `DeepAnalysisResult`, and supporting models for type-safe deep analysis

### Changed

- **`analyze` command**: New `--deep` flag for full ARIZ-85C analysis; mutually exclusive with `--method`. `--deep-model` and `--reasoning-effort` allow using a reasoning model for Passes 1 & 3 while keeping a cheaper model for Pass 2 pipelines
- **Version**: Bumped to 0.9.0

## [0.8.0] - 2026-03-15

### Added

- **Multi-tool TRIZ analysis**: Auto-classifies problems and routes to the best TRIZ tool — technical contradictions, physical contradictions, Su-Field analysis, function analysis, trimming, or evolution trends
- **Problem router**: Classifier + IFR formulation + root cause analysis fallback; `--method` flag to force a specific tool, `--router-model` to use a different model for classification
- **Physical contradiction pipeline**: Identifies opposing requirements on a single property and recommends separation principles (time, space, scale, condition)
- **Su-Field analysis pipeline**: Models substance-field interactions, classifies problem type (incomplete/harmful/inefficient), recommends from 55 standard solutions
- **Function analysis pipeline**: Decomposes system into components and functions, identifies harmful/insufficient/excessive functions with recommendations
- **Trimming pipeline**: Identifies components that can be removed and shows how their functions are redistributed to remaining components
- **Trends pipeline**: Positions technology on 8 TRIZ evolution trends with system operator (9-screen) framing, predicts next evolutionary stages
- **Ideal Final Result (IFR)**: Every analysis now starts with IFR formulation — the ideal breakthrough target
- **Root cause analysis**: Automatically triggered when classifier confidence is low (< 0.4) to reformulate vague problems before routing
- **TRIZ knowledge data**: 4 separation principles, 55 standard solutions (5 classes), 8 evolution trends with stages
- **Knowledge loaders**: `separation.py`, `solutions.py`, `trends.py` following the same `@lru_cache` pattern as existing loaders

### Changed

- **`AnalysisResult` model**: Now tool-agnostic with `method`, `method_confidence`, `secondary_method`, `ideal_final_result`, and `details` dict; contradiction-specific fields kept for backward compatibility
- **`analyze` command**: Routes through the problem classifier by default; add `--method` to force a specific tool and `--router-model` to override the classification model
- **CLI output**: Common header (problem, method, IFR) + method-specific renderer + common patent table + solution directions; tip line suggests secondary method
- **Config**: Added `llm.router_model` (defaults to `classify_model`)
- **Version**: Bumped to 0.8.0

## [0.7.0] - 2026-03-15

### Added

- **Patent assignee field**: `assignee` column in patents table and Patent model — engineers see which companies solved similar problems
- **Hybrid patent search**: `search_patents_hybrid()` combines vector similarity with TRIZ principle overlap and contradiction matching for more relevant patent results
- **Solution directions**: `analyze` now generates 2-3 concrete, actionable solution directions that apply recommended TRIZ principles to the user's specific problem
- **Enriched patent examples**: `analyze` output now includes assignee, filing date, and matched TRIZ principles for each patent
- **Source patent in ideas**: `discover` ideas now reference the patent that inspired them via `source_patent_id`

### Changed

- **Analyze pipeline**: Uses hybrid search instead of pure vector similarity; adds a solution directions LLM call after patent search
- **Discover pipeline**: Passes patent titles, assignees, and IDs as context to idea generation prompt
- **CLI**: Patent examples in `analyze` rendered as Rich table (ID, Assignee, Title, Matched Principles); solution directions shown as numbered list; discover ideas table includes Source Patent column
- **Seed dataset**: `battery_patents.json` now includes `assignee` field (null for existing entries)

### Note

- Existing users need `triz-ai init --force` to pick up the new `assignee` column (no DB migrations)

## [0.6.0] - 2026-03-15

### Added

- **PyPI publishing**: `pip install triz-ai` now works — TRIZ data files bundled inside the package
- **GitHub CI workflow**: Test on push/PR, publish to PyPI on release or manual dispatch
- **`python-dotenv` dependency**: Now declared in `pyproject.toml` (was used but missing)
- **PyPI metadata**: Authors, license, classifiers, project URLs

### Changed

- **Zero-setup analyze**: `triz-ai analyze "problem"` works immediately without running `init` first; shows a tip to ingest patents when no patent examples are found
- **`init` command**: Now documented as only needed with `--force` to reset the database
- **Python >=3.12**: Lowered from >=3.14 for broader compatibility
- **TRIZ data location**: Moved from `data/triz/` to `src/triz_ai/data/` so data is included in pip installs

## [0.5.0] - 2026-03-15

### Added

- **Separate classify model**: Classification uses a smaller, faster model (`openrouter/nvidia/nemotron-3-nano-30b-a3b:free` by default) — configurable via `llm.classify_model` in config or `--classify-model` CLI flag on `ingest`

### Changed

- **Auto-classify on ingest**: Patents are now automatically classified through the TRIZ lens during ingestion — no separate classify step needed. The user journey simplifies from `init → ingest → classify → analyze/discover` to `init → ingest → analyze/discover`.
- **Removed `classify` CLI command**: Classification is now internal-only (called by `ingest` and available as an engine function). The `classify()` engine function remains for programmatic use.
- **Simplified `evolve` pipeline**: Removed the "batch classify unclassified patents" step from both `run_evolution()` and `run_parameter_evolution()` — patents arrive pre-classified from ingestion.
- **`matrix seed` documented as power-user command**: The matrix ships pre-seeded at 95.7% fill rate, so seeding is rarely needed.

## [0.4.0] - 2026-03-15

### Added

- **Hybrid matrix extension**: LLM-seeds missing contradiction matrix cells for parameters 40-50, boosting fill rate from 59.4% to 95.7% (~890 new cells)
- **`triz-ai matrix seed`**: One-time batch operation to populate missing matrix cells via LLM; `--force` to re-seed all 40-50 cells
- **`triz-ai matrix stats`**: Show matrix fill rate, patent observation counts, and top observed parameter pairs
- **Patent-observed matrix refinement**: Classifications now auto-record matrix observations; over time, empirical data from ≥3 patents refines/overrides LLM-seeded entries
- **`lookup_with_observations()`**: Merged lookup that scores principles by observation count with a bonus for static-matrix agreement, falling back to static-only when no store is available
- **`matrix_observations` DB table**: Stores (improving, worsening, principle_id, patent_id, confidence) associations from classified patents
- **`seed_matrix_prompt()`**: TRIZ-aware prompt template for matrix seeding with parameter lists, principle lists, and example rows

### Changed

- **Analyzer**: Now uses `lookup_with_observations()` for principle recommendations, merging static matrix with patent data when a store is available
- **Classifier**: Automatically records matrix observations on every classification that has a store and patent_id

## [0.3.0] - 2026-03-14

### Added

- **Progress bars**: Rich progress indicators for patent ingestion and evolution pipeline classification steps
- **User-friendly error messages**: Actionable guidance for common failures — authentication, rate limits, timeouts, connection errors, model not found
- **Auto-classify in discover**: `discover` command now automatically classifies unclassified patents in the domain before generating insights
- **Fuzzy domain matching**: `discover` and domain queries now match patents by domain field, title, or abstract (case-insensitive contains), so txt-ingested patents without a domain field are included

### Fixed

- **Suppressed litellm console noise**: No more red `Provider List` and `Give Feedback` lines polluting CLI output
- **Smart retry logic**: Only retries LLM calls on validation/parsing errors; auth, network, and rate limit errors are raised immediately with helpful messages instead of pointlessly retrying

## [0.2.0] - 2026-03-14

### Added

- **Prompt engineering**: Context-aware prompts that inject relevant TRIZ knowledge (principles, parameters, matrix data) into system prompts, keeping token budget under ~2K tokens per call
- **Free default models**: Both LLM and embeddings now default to free OpenRouter models — `stepfun/step-3.5-flash:free` for completions, `nvidia/llama-nemotron-embed-vl-1b-v2:free` for embeddings
- **Configurable embedding dimensions**: `embeddings.dimensions` config option (default 768) passed to embedding API calls
- **`.env` support**: API keys loaded from `.env` file via python-dotenv

### Changed

- **Python 3.14**: Bumped minimum Python version from 3.11 to 3.14 (latest stable)
- **Default LLM model**: Changed from `openrouter/google/gemini-2.5-flash` to `openrouter/stepfun/step-3.5-flash:free`
- **Default embedding model**: Changed from `ollama/nomic-embed-text` to `openrouter/nvidia/llama-nemotron-embed-vl-1b-v2:free` — eliminates Ollama dependency for getting started
- **LLM client**: `_complete()` method now uses `TypeVar` for proper generic return types (caught by ty type checker)

## [0.1.0] - 2026-03-14

### Added

- **CLI** (`triz-ai`): Typer-based CLI with 6 commands — `analyze`, `classify`, `discover`, `evolve`, `ingest`, `init`
- **Output formats**: All commands support `--format text|json|markdown` and `--model` override
- **TRIZ knowledge base**: 40 inventive principles, 39 engineering parameters, full 39x39 asymmetric contradiction matrix loaded from JSON data files
- **LLM client**: litellm wrapper with pydantic response validation and single-retry logic for malformed responses; supports any litellm-compatible model string
- **Patent store**: SQLite + sqlite-vec storage with vector similarity search (768-dim embeddings), patent CRUD, classification storage, and candidate principle tracking
- **Patent ingestion**: Support for `.txt`, `.pdf` (via pdfplumber), and `.json` batch formats with automatic embedding
- **Analysis engine**: Problem → contradiction extraction → matrix lookup → patent search → solution directions
- **Patent classifier**: Classify patents by TRIZ principles with confidence scoring
- **Discovery engine**: Aggregate principle usage by domain, identify underused principles, generate novel ideas
- **Idea evaluator**: Score ideas against prior art using vector similarity
- **Evolution pipeline**: Batch classify → filter low-confidence → LLM-based semantic clustering → propose candidate new principles
- **Interactive review**: Rich-formatted terminal UI for accepting/rejecting candidate principles
- **Configuration**: YAML config at `~/.triz-ai/config.yaml` with sensible defaults (model, embeddings, database path, evolution threshold)
- **Test suite**: 53 unit tests covering all modules with mocked LLM calls
- **Pre-commit hooks**: ruff (lint/format), ty (type check), validate-pyproject, security checks, fast pytest on pre-push
- **Sample data**: 5 sample patent fixtures (3 txt, 1 JSON batch with 3 patents) for testing and demos

[0.12.0]: https://github.com/flyersworder/triz-ai/compare/v0.11.1...v0.12.0
[0.11.1]: https://github.com/flyersworder/triz-ai/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/flyersworder/triz-ai/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/flyersworder/triz-ai/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/flyersworder/triz-ai/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/flyersworder/triz-ai/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/flyersworder/triz-ai/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/flyersworder/triz-ai/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/flyersworder/triz-ai/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/flyersworder/triz-ai/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/flyersworder/triz-ai/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/flyersworder/triz-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/flyersworder/triz-ai/releases/tag/v0.1.0
