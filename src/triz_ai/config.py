"""Configuration loading with pydantic-settings."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# Load .env from project root (if present) so API keys are available to litellm
load_dotenv()


class LLMConfig(BaseModel):
    default_model: str = "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
    classify_model: str = "openrouter/nvidia/nemotron-3-nano-30b-a3b:free"
    api_base: str | None = None  # Custom API base URL (e.g., litellm proxy)
    api_key: str | None = None  # Custom API key (overrides env var)


class EmbeddingsConfig(BaseModel):
    model: str = "openrouter/nvidia/llama-nemotron-embed-vl-1b-v2:free"
    dimensions: int = 768
    api_base: str | None = None  # Custom API base URL for embeddings
    api_key: str | None = None  # Custom API key for embeddings


class DatabaseConfig(BaseModel):
    path: str = "~/.triz-ai/patents.db"


class EvolutionConfig(BaseModel):
    auto_classify: bool = True
    review_threshold: float = 0.7


class Settings(BaseSettings):
    llm: LLMConfig = LLMConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    database: DatabaseConfig = DatabaseConfig()
    evolution: EvolutionConfig = EvolutionConfig()


def load_config() -> Settings:
    """Load config from ~/.triz-ai/config.yaml, falling back to defaults."""
    config_path = Path.home() / ".triz-ai" / "config.yaml"
    if config_path.exists():
        import yaml

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return Settings(**data)
    return Settings()


def get_db_path() -> Path:
    """Get resolved database path."""
    settings = load_config()
    return Path(settings.database.path).expanduser()
