# triz-ai

AI-powered TRIZ innovation engine — analyze technical problems, classify patents, and discover new inventive principles.

## What is this?

`triz-ai` combines [TRIZ](https://en.wikipedia.org/wiki/TRIZ) (Theory of Inventive Problem Solving) with AI and real patent data. It goes beyond static TRIZ tools by using AI to discover candidate new principles from modern patents, continuing Altshuller's original work.

- **Patent-grounded** — every suggestion is backed by real patent evidence
- **Evolving principles** — discovers candidate new principles from modern patents
- **Provider-agnostic** — works with OpenRouter, Ollama, Anthropic, OpenAI, and 100+ providers via litellm
- **Zero infrastructure** — local SQLite database with built-in vector search

## Installation

```bash
# Requires Python 3.11+ and uv
uv sync
```

Set up your LLM provider API key:

```bash
export OPENROUTER_API_KEY="your-key"  # default provider
# or ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.
```

For embeddings, the default uses Ollama locally:

```bash
ollama pull nomic-embed-text
```

## Quick Start

```bash
# Initialize the database
uv run triz-ai init

# Ingest sample patents
uv run triz-ai ingest tests/fixtures/sample_patents/

# Analyze a technical problem
uv run triz-ai analyze "How to make an EV battery charge faster without overheating"

# Classify a patent through TRIZ lens
uv run triz-ai classify tests/fixtures/sample_patents/battery_thermal.txt

# Discover underused principles in a domain
uv run triz-ai discover --domain "battery technology"

# Run evolution pipeline to find candidate new principles
uv run triz-ai evolve
uv run triz-ai evolve --review  # interactive accept/reject
```

## Commands

| Command | Description |
|---------|-------------|
| `analyze` | Full TRIZ pipeline: extract contradiction → matrix lookup → patent search → solution directions |
| `classify` | Classify a patent by TRIZ principles (accepts file, text, or stdin) |
| `discover` | Find underused principles in a domain and generate novel ideas |
| `evolve` | Discover candidate new TRIZ principles from patent clusters |
| `ingest` | Load patents from .txt, .pdf, or .json files |
| `init` | Initialize (or recreate with `--force`) the patent database |

All commands support `--format text|json|markdown` and `--model` to override the LLM model.

## Project Structure

```
src/triz_ai/
  cli.py              # Typer CLI entry point
  config.py            # Config loading (~/.triz-ai/config.yaml)
  knowledge/           # 40 principles, 39 parameters, 39x39 matrix
  engine/              # analyzer, classifier, generator, evaluator
  patents/             # SQLite + sqlite-vec store, ingestion pipeline
  evolution/           # Candidate principle discovery + review
  llm/                 # litellm wrapper with pydantic validation
data/triz/             # Static TRIZ JSON data files
tests/                 # 53 unit tests
```

## Configuration

Config lives at `~/.triz-ai/config.yaml`:

```yaml
llm:
  default_model: openrouter/google/gemini-2.5-flash

embeddings:
  model: ollama/nomic-embed-text

database:
  path: ~/.triz-ai/patents.db

evolution:
  review_threshold: 0.7
```

Any [litellm-supported model string](https://docs.litellm.ai/docs/providers) works — just change the model and set the corresponding API key.

## Development

```bash
uv sync                              # Install dependencies
uv run pytest                        # Run tests
uv run pre-commit install            # Install git hooks
uv run pre-commit install --hook-type pre-push
```

## License

MIT
