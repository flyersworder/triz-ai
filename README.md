# triz-ai

[![PyPI](https://img.shields.io/pypi/v/triz-ai.svg)](https://pypi.org/project/triz-ai/)
[![License](https://img.shields.io/pypi/l/triz-ai.svg)](https://github.com/flyersworder/triz-ai/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/triz-ai.svg)](https://pypi.org/project/triz-ai/)

AI-powered TRIZ innovation engine — analyze technical problems, classify patents, and discover new inventive principles.

## What is this?

`triz-ai` combines [TRIZ](https://en.wikipedia.org/wiki/TRIZ) (Theory of Inventive Problem Solving) with AI and real patent data. It goes beyond static TRIZ tools by using AI to discover candidate new principles from modern patents, continuing Altshuller's original work.

- **Patent-grounded** — every suggestion is backed by real patent evidence
- **50 engineering parameters** — extends Altshuller's 39 with modern domains (security, sustainability, scalability, etc.)
- **Evolving principles & parameters** — discovers candidate new principles and parameters from modern patents
- **Provider-agnostic** — works with OpenRouter, Ollama, Anthropic, OpenAI, and 100+ providers via litellm
- **Zero infrastructure** — local SQLite database with built-in vector search

## Installation

```bash
pip install triz-ai
```

Or for development:

```bash
# Requires Python 3.12+ and uv
uv sync
```

Set up your LLM provider API key:

```bash
export OPENROUTER_API_KEY="your-key"
```

Or use a `.env` file: `echo 'OPENROUTER_API_KEY=your-key' > .env`

## Quick Start

```bash
# Analyze a technical problem (works immediately — no setup needed)
triz-ai analyze "How to make an EV battery charge faster without overheating"

# For patent-backed examples, ingest patent data
triz-ai ingest data/patents/battery_patents.json

# Discover underused principles in a domain
triz-ai discover --domain "battery technology"

# Run evolution pipeline to find candidate new principles
triz-ai evolve
triz-ai evolve --review  # interactive accept/reject

# Discover candidate new engineering parameters
triz-ai evolve --parameters
triz-ai evolve --parameters --review

# View matrix statistics
triz-ai matrix stats
```

## Commands

| Command | Description |
|---------|-------------|
| `analyze` | Full TRIZ pipeline: extract contradiction → matrix lookup → patent search → solution directions |
| `discover` | Find underused principles in a domain and generate novel ideas |
| `evolve` | Discover candidate new TRIZ principles (`--parameters` for parameters) |
| `ingest` | Ingest and auto-classify patents from .txt, .pdf, or .json files |
| `init` | Reset the patent database (only needed with `--force`) |
| `matrix seed` | LLM-seed missing matrix cells for params 40-50 (power-user) |
| `matrix stats` | Show matrix fill rate and patent observation statistics |

All commands support `--format text|json|markdown` and `--model` to override the LLM model.

## Project Structure

```
src/triz_ai/
  cli.py              # Typer CLI entry point
  config.py            # Config loading (~/.triz-ai/config.yaml)
  data/                # Static TRIZ JSON data files (principles, parameters, matrix)
  knowledge/           # 40 principles, 50 parameters, contradiction matrix
  engine/              # analyzer, classifier, generator, evaluator
  patents/             # SQLite + sqlite-vec store, ingestion pipeline
  evolution/           # Candidate principle & parameter discovery + review
  llm/                 # litellm wrapper with pydantic validation
tests/                 # 72 unit tests
```

## Configuration

Config lives at `~/.triz-ai/config.yaml`:

```yaml
llm:
  default_model: openrouter/nvidia/nemotron-3-super-120b-a12b:free
  classify_model: openrouter/nvidia/nemotron-3-nano-30b-a3b:free  # smaller model for classification

embeddings:
  model: openrouter/nvidia/llama-nemotron-embed-vl-1b-v2:free
  dimensions: 768

database:
  path: ~/.triz-ai/patents.db

evolution:
  review_threshold: 0.7
```

Any [litellm-supported model string](https://docs.litellm.ai/docs/providers) works — just change the model and set the corresponding API key.

### Using a custom LLM gateway (e.g., company litellm proxy)

```yaml
llm:
  default_model: gpt-4o
  api_base: https://llm-proxy.internal/v1
  api_key: your-proxy-token

embeddings:
  model: text-embedding-3-small
  dimensions: 768
  api_base: https://llm-proxy.internal/v1
  api_key: your-proxy-token
```

## Development

```bash
uv sync                              # Install dependencies
uv run pytest                        # Run tests
uv run pre-commit install            # Install git hooks
uv run pre-commit install --hook-type pre-push
```

## License

MIT
