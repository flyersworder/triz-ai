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

`src/triz_ai/` modules: `cli.py` (Typer CLI) → `engine/` (analyzer, classifier, generator, evaluator) → `llm/client.py` (litellm wrapper) → `patents/` (SQLite + sqlite-vec store, ingestion) → `knowledge/` (TRIZ data from `data/triz/*.json`) → `evolution/` (candidate principle discovery).

## Key Constraints

- **Contradiction matrix is asymmetric** — improving A worsening B ≠ improving B worsening A
- **Embedding dimension is 768** — changing embedding model requires `triz-ai init --force`
- **LLM responses validated via pydantic** — malformed → 1 retry with stricter prompt, then fail
- **No DB migrations** — schema changes require `triz-ai init --force`
- **Token budget** — only inject relevant matrix rows/principles into prompts, not full dataset

## References

- Design spec: `docs/specs/2026-03-14-triz-ai-design.md`
- Config: `~/.triz-ai/config.yaml` (defaults: model `openrouter/stepfun/step-3.5-flash:free`, embeddings `openrouter/nvidia/llama-nemotron-embed-vl-1b-v2:free`)
- API keys: `.env` file (loaded via python-dotenv) or env vars per litellm conventions (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, etc.)
