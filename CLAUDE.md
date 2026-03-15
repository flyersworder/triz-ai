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

`src/triz_ai/` modules: `cli.py` (Typer CLI) → `engine/router.py` (problem classifier + IFR + RCA + dispatch) → `engine/` (analyzer, physical, su_field, function_analysis, trimming, trends, classifier, generator, evaluator) → `llm/client.py` (litellm wrapper) → `patents/` (SQLite + sqlite-vec store, ingestion, matrix observations) → `knowledge/` (TRIZ data from `src/triz_ai/data/*.json`, `matrix_builder.py` for LLM-seeding) → `evolution/` (candidate principle and parameter discovery).

### Multi-Tool Routing

`analyze` auto-classifies problems and routes to the best TRIZ pipeline:
- `technical_contradiction` → `analyzer.py` (improve X without worsening Y)
- `physical_contradiction` → `physical.py` (part must be A AND B)
- `su_field` → `su_field.py` (detection/measurement/interaction problems)
- `function_analysis` → `function_analysis.py` (harmful component interactions)
- `trimming` → `trimming.py` (simplification/cost reduction)
- `trends` → `trends.py` (technology evolution + system operator)

IFR is always formulated first. If classifier confidence < 0.4, RCA reformulates before re-routing. `--method` flag bypasses classifier. `--router-model` overrides classification model.

## Key Constraints

- **6 TRIZ analysis methods** — technical contradiction, physical contradiction, Su-Field, function analysis, trimming, trends. Router auto-classifies; `--method` forces one.
- **50 engineering parameters** — IDs 1-39 are Altshuller's originals, 40-50 are modern extensions (Mann's Matrix 2010). The static contradiction matrix covers 1-39; cells for 40-50 can be LLM-seeded via `triz-ai matrix seed` and refined by patent observations over time. `lookup_with_observations()` merges both sources.
- **Contradiction matrix is asymmetric** — improving A worsening B ≠ improving B worsening A
- **Embedding dimension is 768** — changing embedding model requires `triz-ai init --force`
- **LLM responses validated via pydantic** — malformed → 1 retry with stricter prompt, then fail
- **Auto-init** — `analyze` and other commands work without running `init` first; `init` is only needed with `--force` to reset the database
- **Hybrid patent search** — `analyze` (technical contradiction) uses `search_patents_hybrid()` which combines vector similarity with principle overlap bonus (0.3/principle, cap 0.6) and contradiction match bonus (0.4 exact, 0.2 partial). Other methods use vector-only search.
- **No DB migrations** — schema changes require `triz-ai init --force`
- **Token budget** — only inject relevant matrix rows/principles into prompts, not full dataset
- **TRIZ knowledge data** — `separation_principles.json` (4 categories), `standard_solutions.json` (76 solutions, 5 classes), `evolution_trends.json` (8 trends with stages)

## Models

- Default LLM: `openrouter/nvidia/nemotron-3-super-120b-a12b:free`
- Default classify model: `openrouter/nvidia/nemotron-3-nano-30b-a3b:free` (smaller model used for patent classification during ingest)
- Default embeddings: `openrouter/nvidia/llama-nemotron-embed-vl-1b-v2:free`
- Alternative classify model: `openrouter/google/gemini-3.1-flash-lite-preview` — not free but extremely cheap ($0.25/$1.50 per M tokens), works within OpenRouter's default free testing allowance. Use via `--classify-model` flag or `llm.classify_model` in config. `classify_patent()` sets `max_tokens=1024` to avoid reserving the full 65K output window against credits.

## References

- Design spec: `docs/specs/2026-03-14-triz-ai-design.md`
- Config: `~/.triz-ai/config.yaml`
- API keys: `.env` file (loaded via python-dotenv) or env vars per litellm conventions (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, etc.)
