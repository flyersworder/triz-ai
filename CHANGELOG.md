# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/flyersworder/triz-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/flyersworder/triz-ai/releases/tag/v0.1.0
