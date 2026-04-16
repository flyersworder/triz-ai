# triz-ai

[![PyPI](https://img.shields.io/pypi/v/triz-ai.svg)](https://pypi.org/project/triz-ai/)
[![Tests](https://github.com/flyersworder/triz-ai/actions/workflows/ci-and-publish.yml/badge.svg)](https://github.com/flyersworder/triz-ai/actions/workflows/ci-and-publish.yml)
[![License](https://img.shields.io/pypi/l/triz-ai.svg)](https://github.com/flyersworder/triz-ai/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/triz-ai.svg)](https://pypi.org/project/triz-ai/)

AI-powered TRIZ innovation engine — analyze technical problems, classify patents, and discover new inventive principles.

## What is this?

`triz-ai` combines [TRIZ](https://en.wikipedia.org/wiki/TRIZ) (Theory of Inventive Problem Solving) with AI and real patent data. It goes beyond static TRIZ tools by using AI to discover candidate new principles from modern patents, continuing Altshuller's original work.

- **Patent-grounded** — every suggestion is backed by real patent evidence, with assignee and filing date
- **Hybrid search** — finds relevant patents using both vector similarity and TRIZ principle/contradiction matching
- **Solution directions** — generates concrete, actionable solution approaches, not just abstract principles
- **50 engineering parameters** — extends Altshuller's 39 with modern domains (security, sustainability, scalability, etc.)
- **Self-evolving** — learns from web search results during analysis; discovers candidate new principles from usage patterns and patents
- **Provider-agnostic** — works with OpenRouter, Ollama, Anthropic, OpenAI, and 100+ providers via litellm
- **Pluggable storage** — local SQLite by default; swap to Postgres, DynamoDB, etc. via the `PatentRepository` protocol
- **Zero infrastructure** — works out of the box with local SQLite + built-in vector search

## Installation

```bash
# With a self-hosted litellm gateway or OpenAI-compatible endpoint:
pip install triz-ai

# For direct access to any provider (OpenRouter, Anthropic, Google, Ollama, etc.):
pip install triz-ai[litellm]
```

Or for development:

```bash
# Requires Python 3.12+ and uv
uv sync
```

Set up your LLM provider:

```bash
# Option 1: Direct provider access (requires litellm extra)
export OPENROUTER_API_KEY="your-key"

# Option 2: Self-hosted litellm gateway (no extra needed)
# Set api_base in ~/.triz-ai/config.yaml:
#   llm:
#     api_base: http://your-litellm-gateway:4000
```

Or use a `.env` file: `echo 'OPENROUTER_API_KEY=your-key' > .env`

## Quick Start

```bash
# Analyze a problem — auto-classifies and routes to the best TRIZ tool
triz-ai analyze "How to increase SiC MOSFET switching speed without increasing EMI"
# → Routes to: technical contradiction analysis

triz-ai analyze "The solder joint must be rigid for reliability but flexible for thermal cycling"
# → Routes to: physical contradiction analysis

triz-ai analyze "How to detect delamination without adding sensors"
# → Routes to: Su-Field analysis

triz-ai analyze "The adhesive layer damages the silicon die during thermal cycling"
# → Routes to: function analysis

triz-ai analyze "Reduce the BOM cost of this gate driver circuit"
# → Routes to: trimming analysis

triz-ai analyze "What is the next generation of SiC packaging technology?"
# → Routes to: trends analysis

# Deep ARIZ-85C analysis — reformulates, runs multiple tools, verifies against IFR
triz-ai analyze "How to increase SiC MOSFET switching speed without increasing EMI" --deep

# Deep with a reasoning model for reformulation/synthesis, cheaper model for pipelines
triz-ai analyze "problem" --deep \
  --model "openrouter/nvidia/nemotron-3-super-120b-a12b:free" \
  --deep-model "openrouter/deepseek/deepseek-r1:free" \
  --reasoning-effort high

# Force a specific method
triz-ai analyze "How to detect delamination" --method su-field

# For patent-backed examples, ingest patent data
triz-ai ingest data/patents/battery_patents.json

# Discover underused principles in a domain
triz-ai discover --domain "battery technology"

# Run evolution pipeline to find candidate new principles
triz-ai evolve
triz-ai evolve --review  # interactive accept/reject

# Consolidate web search observations into matrix data
triz-ai consolidate

# View matrix statistics
triz-ai matrix stats
```

## Commands

| Command | Description |
|---------|-------------|
| `analyze` | Auto-routes to the best TRIZ tool (6 methods); `--method` to force one; `--deep` for full ARIZ-85C |
| `consolidate` | Consolidate web search observations into matrix data and candidate principles |
| `discover` | Find underused principles in a domain and generate patent-grounded ideas |
| `evolve` | Discover candidate new TRIZ principles (`--parameters` for parameters) |
| `ingest` | Ingest and auto-classify patents from .txt, .pdf, or .json files |
| `init` | Reset the patent database (only needed with `--force`) |
| `matrix seed` | LLM-seed missing matrix cells for params 40-50 (power-user) |
| `matrix stats` | Show matrix fill rate and patent observation statistics |

### Analysis Methods

| Method | When to use |
|--------|-------------|
| `technical-contradiction` | Improving X worsens Y |
| `physical-contradiction` | A part must be both A and B |
| `su-field` | Detection/measurement problems |
| `function-analysis` | A component causes harm |
| `trimming` | Simplify / reduce cost |
| `trends` | Where is this technology going? |

All commands support `--format text|json|markdown` and `--model` to override the LLM model.

### Deep ARIZ-85C Analysis

Use `--deep` for a full 3-pass analysis that reformulates the problem, runs multiple TRIZ tools in parallel, and verifies solutions against the Ideal Final Result:

```bash
# Basic deep analysis
triz-ai analyze "problem" --deep

# Use a reasoning model for reformulation/synthesis, cheaper model for pipelines
triz-ai analyze "problem" --deep \
  --deep-model "openrouter/deepseek/deepseek-r1:free" \
  --reasoning-effort high

# Mix models: cheap for Pass 2 pipelines, reasoning for Passes 1 & 3
triz-ai analyze "problem" --deep \
  --model "openrouter/nvidia/nemotron-3-super-120b-a12b:free" \
  --deep-model "anthropic/claude-sonnet-4-6" \
  --reasoning-effort medium
```

| Flag | Affects | Purpose |
|------|---------|---------|
| `--model` | Pass 2 (pipelines) + fallback | Base model for parallel tool research |
| `--deep-model` | Pass 1 & 3 | Reasoning model for reformulation + synthesis |
| `--reasoning-effort` | Pass 1 & 3 | `low`/`medium`/`high` — litellm translates across providers |

## Research Tools (Programmatic API)

Research tools can participate in three pipeline stages: **context** (before LLM extraction), **search** (during patent search), and **enrichment** (after solution generation):

```python
from triz_ai import ResearchTool
from triz_ai.engine.router import route
from triz_ai.llm.client import LLMClient

# Search-only tool (default, backward compatible)
google_patents = ResearchTool(
    name="google_patents",
    description="Search Google Patents for prior art. Best for recent filings.",
    fn=lambda query, ctx: [  # ctx includes {"stage": "search", ...}
        {"title": f"Patent about {query}", "abstract": "...", "url": "..."}
    ],
)

# Multi-stage tool: provides context before analysis + enrichment after
web_search = ResearchTool(
    name="web_search",
    description="General web search for domain knowledge and feasibility data.",
    fn=lambda query, ctx: (
        [{"content": f"Domain context for: {query}"}] if ctx["stage"] == "context"
        else [{"title": "Feasibility note", "content": "..."}]
    ),
    stages=["context", "enrichment"],
)

# Normal mode: all tools run automatically at their declared stages
result = route("battery energy density", LLMClient(),
               research_tools=[google_patents, web_search])
# result.enrichment contains enrichment-stage results

# Deep mode: LLM selects which tools to use
from triz_ai.engine.ariz import orchestrate_deep
result = orchestrate_deep("battery problem", LLMClient(), store=None,
                          research_tools=[google_patents, web_search])
```

Each tool's `fn(query, context)` receives a search query and a context dict with `{"stage": str}` plus stage-specific data. Return format depends on stage:
- **context**: `[{"content": "..."}]` — text prepended to problem description
- **search**: `[{"title": "...", "abstract": "..."}]` — patent-like results (optional: `"id"`, `"assignee"`, `"filing_date"`, `"url"`, `"matched_principles"`)
- **enrichment**: `[{"title": "...", "content": "..."}]` — stored in `AnalysisResult.enrichment`

## Project Structure

```
src/triz_ai/
  cli.py              # Typer CLI entry point
  config.py            # Config loading (--config / TRIZ_AI_CONFIG / ~/.triz-ai/config.yaml)
  data/                # TRIZ JSON data (principles, parameters, matrix, separation, solutions, trends)
  knowledge/           # Loaders for all TRIZ knowledge data
  engine/              # analyzer, router, ariz orchestrator, 5 pipelines, classifier, generator, evaluator
  patents/             # PatentRepository protocol + SQLite default, vector search, ingestion
  evolution/           # Candidate principle & parameter discovery, review, and self-evolution
  llm/                 # litellm wrapper with pydantic validation
tests/                 # 227 unit tests
```

## Configuration

Config is loaded from (highest priority first):

1. `--config` CLI flag — `triz-ai --config /path/to/config.yaml analyze "..."`
2. `TRIZ_AI_CONFIG` environment variable — `export TRIZ_AI_CONFIG=/app/config.yaml`
3. `~/.triz-ai/config.yaml` (default)

Example:

```yaml
llm:
  default_model: openrouter/nvidia/nemotron-3-super-120b-a12b:free
  classify_model: openrouter/nvidia/nemotron-3-nano-30b-a3b:free
  # router_model: null           # model for problem classification (defaults to classify_model)
  # deep_model: null             # model for ARIZ deep passes 1 & 3 (defaults to default_model)
  # reasoning_effort: null       # low/medium/high for reasoning models in deep mode

embeddings:
  model: openrouter/nvidia/llama-nemotron-embed-vl-1b-v2:free
  dimensions: 768

database:
  path: ~/.triz-ai/patents.db

evolution:
  review_threshold: 0.7
  consolidation_interval: 25       # auto-consolidate every N analyses
  retention_days: 180              # prune consolidated observations after N days
  source_confidence_weight: 0.6    # web results confidence discount vs patents
```

Any [litellm-supported model string](https://docs.litellm.ai/docs/providers) works — just change the model and set the corresponding API key.

### Using a custom LLM gateway (e.g., company litellm proxy)

```yaml
llm:
  default_model: gpt-4o              # base model for analysis pipelines
  classify_model: gpt-4o-mini        # smaller model for patent classification during ingest
  deep_model: o3                     # reasoning model for ARIZ deep passes 1 & 3
  reasoning_effort: medium           # low/medium/high for the deep model
  api_base: https://llm-proxy.internal/v1
  api_key: your-proxy-token
  ssl_verify: false                # disable for corporate proxies with internal CA certs

embeddings:
  model: text-embedding-3-small
  dimensions: 768
  api_base: https://llm-proxy.internal/v1
  api_key: your-proxy-token
```

### Environment variable interpolation

Config YAML values support shell-style `${VAR}` and `${VAR:-default}` substitution. Resolution happens once when the config file is first read, before pydantic validation. This is the recommended way to inject API keys in containerized deployments (Kubernetes Secrets, OpenShift, Docker Compose), where the YAML is baked into the image and secrets arrive as environment variables.

```yaml
llm:
  api_base: "${LITELLM_GATEWAY_URL:-https://openrouter.ai/api/v1}"
  api_key: "${LITELLM_MASTER_KEY}"

embeddings:
  api_base: "${LITELLM_GATEWAY_URL:-https://openrouter.ai/api/v1}"
  api_key: "${LITELLM_MASTER_KEY}"
```

Rules:

- `${VAR}` — fails at startup if `VAR` is unset or empty. Use this for required secrets so missing config breaks loudly instead of sending empty auth headers.
- `${VAR:-default}` — shell `:-` semantics: both unset and empty env vars fall back to `default`. Use for optional fields like `api_base` with a sensible production default.
- `${VAR:-}` — explicit opt-in for empty/unset; yields the empty string. Useful when a field is genuinely optional and an absent value is preferable to a non-empty default.
- `$$` — escape a literal `$`. For example, `$${FOO}` renders as the literal string `${FOO}`.
- Nested tokens (`${FOO_${BAR}}`) are not supported; compose in the shell before starting the process.

You can also override models per-command:

```bash
triz-ai ingest data/ --classify-model gpt-4o-mini
triz-ai analyze "problem" --deep --deep-model o3 --reasoning-effort high
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
