"""Configuration loading with pydantic-settings."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# Load .env from project root (if present) so API keys are available to litellm
load_dotenv()


class ConfigError(Exception):
    """Raised when config loading or env var interpolation fails."""


_NAME_START = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_"
_NAME_CONT = _NAME_START + "0123456789"


def _resolve_tokens(value: str, field_path: str) -> str:
    """Resolve ${VAR} and ${VAR:-default} tokens in a string value.

    See docs/specs/2026-04-16-env-var-interpolation-design.md for the grammar.
    Raises ConfigError on unclosed tokens, empty/invalid names, or
    unset/empty vars without defaults.
    """
    out: list[str] = []
    i = 0
    n = len(value)
    while i < n:
        # Escape: $$ -> $
        if value[i] == "$" and i + 1 < n and value[i + 1] == "$":
            out.append("$")
            i += 2
            continue
        # Token: ${...}
        if value[i] == "$" and i + 1 < n and value[i + 1] == "{":
            i += 2  # past '${'
            name_start = i
            if i < n and value[i] in _NAME_START:
                i += 1
                while i < n and value[i] in _NAME_CONT:
                    i += 1
            name = value[name_start:i]
            if not name:
                if i >= n:
                    raise ConfigError(
                        f"Config field {field_path}: unclosed '${{' in value {value!r}"
                    )
                if value[i] == "}" or value[i : i + 2] == ":-":
                    raise ConfigError(
                        f"Config field {field_path}: empty variable name in {value!r}"
                    )
                raise ConfigError(
                    f"Config field {field_path}: invalid character {value[i]!r} "
                    f"in variable name; {value!r}"
                )
            default: str | None = None
            if i >= n:
                raise ConfigError(f"Config field {field_path}: unclosed '${{' in value {value!r}")
            if value[i] == "}":
                i += 1  # past '}'
            elif value[i : i + 2] == ":-":
                i += 2  # past ':-'
                end = value.find("}", i)
                if end == -1:
                    raise ConfigError(
                        f"Config field {field_path}: unclosed '${{' in value {value!r}"
                    )
                default = value[i:end]
                i = end + 1  # past '}'
            else:
                raise ConfigError(
                    f"Config field {field_path}: invalid character {value[i]!r} "
                    f"in variable name; {value!r}"
                )
            env_val = os.environ.get(name)
            if env_val:
                out.append(env_val)
            elif default is not None:
                out.append(default)
            else:
                raise ConfigError(
                    f"Config field {field_path}: environment variable {name} is not set "
                    f"(or is empty). Either set it to a non-empty value, or provide a "
                    f"default: ${{{name}:-}}"
                )
            continue
        out.append(value[i])
        i += 1
    return "".join(out)


class LLMConfig(BaseModel):
    default_model: str = "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
    classify_model: str = "openrouter/nvidia/nemotron-3-nano-30b-a3b:free"
    router_model: str | None = None  # Defaults to classify_model
    deep_model: str | None = None  # Model for ARIZ deep passes 1 & 3 (defaults to default_model)
    reasoning_effort: str | None = None  # low/medium/high for reasoning models in deep mode
    api_base: str | None = None  # Custom API base URL (e.g., litellm proxy)
    api_key: str | None = None  # Custom API key (overrides env var)
    ssl_verify: bool = True  # Set to false for corporate proxies with internal CA certs


class EmbeddingsConfig(BaseModel):
    model: str = "openrouter/nvidia/llama-nemotron-embed-vl-1b-v2:free"
    dimensions: int = 768
    api_base: str | None = None  # Custom API base URL for embeddings
    api_key: str | None = None  # Custom API key for embeddings


class DatabaseConfig(BaseModel):
    path: str = "~/.triz-ai/patents.db"
    backend: str = "sqlite"
    vector_backend: str = "sqlite-vec"
    vector_options: dict = {}


class EvolutionConfig(BaseModel):
    auto_classify: bool = True
    review_threshold: float = 0.7
    consolidation_interval: int = 25
    retention_days: int = 180
    source_confidence_weight: float = 0.6


class Settings(BaseSettings):
    llm: LLMConfig = LLMConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    database: DatabaseConfig = DatabaseConfig()
    evolution: EvolutionConfig = EvolutionConfig()


_config_path_override: Path | None = None


def set_config_path(path: str | Path | None) -> None:
    """Set a global config path override (used by CLI --config flag)."""
    global _config_path_override
    _config_path_override = Path(path) if path is not None else None


def load_config(config_path: str | Path | None = None) -> Settings:
    """Load config from YAML, falling back to defaults.

    Resolution order:
    1. Explicit ``config_path`` argument (highest priority)
    2. Module-level override set via :func:`set_config_path` (CLI ``--config``)
    3. ``TRIZ_AI_CONFIG`` environment variable
    4. ``~/.triz-ai/config.yaml`` (default)
    """
    if config_path is None:
        config_path = _config_path_override
    if config_path is None:
        env_path = os.environ.get("TRIZ_AI_CONFIG")
        if env_path:
            config_path = Path(env_path)
    if config_path is None:
        config_path = Path.home() / ".triz-ai" / "config.yaml"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        import yaml

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return Settings(**data)
    return Settings()


def get_db_path(config_path: str | Path | None = None) -> Path:
    """Get resolved database path."""
    settings = load_config(config_path)
    return Path(settings.database.path).expanduser()
