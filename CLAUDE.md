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

`src/triz_ai/` modules: `cli.py` (Typer CLI) → `engine/` (analyzer, classifier, generator, evaluator) → `llm/client.py` (litellm wrapper) → `patents/` (SQLite + sqlite-vec store, ingestion, matrix observations) → `knowledge/` (TRIZ data from `src/triz_ai/data/*.json`, `matrix_builder.py` for LLM-seeding) → `evolution/` (candidate principle and parameter discovery).

## Key Constraints

- **50 engineering parameters** — IDs 1-39 are Altshuller's originals, 40-50 are modern extensions (Mann's Matrix 2010). The static contradiction matrix covers 1-39; cells for 40-50 can be LLM-seeded via `triz-ai matrix seed` and refined by patent observations over time. `lookup_with_observations()` merges both sources.
- **Contradiction matrix is asymmetric** — improving A worsening B ≠ improving B worsening A
- **Embedding dimension is 768** — changing embedding model requires `triz-ai init --force`
- **LLM responses validated via pydantic** — malformed → 1 retry with stricter prompt, then fail
- **Auto-init** — `analyze` and other commands work without running `init` first; `init` is only needed with `--force` to reset the database
- **Hybrid patent search** — `analyze` uses `search_patents_hybrid()` which combines vector similarity with principle overlap bonus (0.3/principle, cap 0.6) and contradiction match bonus (0.4 exact, 0.2 partial). Fetches 4x candidates then re-ranks.
- **No DB migrations** — schema changes require `triz-ai init --force`
- **Token budget** — only inject relevant matrix rows/principles into prompts, not full dataset

## Models

- Default LLM: `openrouter/nvidia/nemotron-3-super-120b-a12b:free`
- Default classify model: `openrouter/nvidia/nemotron-3-nano-30b-a3b:free` (smaller model used for patent classification during ingest)
- Default embeddings: `openrouter/nvidia/llama-nemotron-embed-vl-1b-v2:free`
- Alternative classify model: `openrouter/google/gemini-3.1-flash-lite-preview` — not free but extremely cheap ($0.25/$1.50 per M tokens), works within OpenRouter's default free testing allowance. Use via `--classify-model` flag or `llm.classify_model` in config. `classify_patent()` sets `max_tokens=1024` to avoid reserving the full 65K output window against credits.

## References

- Design spec: `docs/specs/2026-03-14-triz-ai-design.md`
- Config: `~/.triz-ai/config.yaml`
- API keys: `.env` file (loaded via python-dotenv) or env vars per litellm conventions (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, etc.)
