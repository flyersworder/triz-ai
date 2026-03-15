# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.5.0]: https://github.com/flyersworder/triz-ai/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/flyersworder/triz-ai/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/flyersworder/triz-ai/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/flyersworder/triz-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/flyersworder/triz-ai/releases/tag/v0.1.0
