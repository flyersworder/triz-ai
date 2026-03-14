"""Configuration loading with pydantic-settings."""

from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class LLMConfig(BaseModel):
    default_model: str = "openrouter/google/gemini-2.5-flash"


class EmbeddingsConfig(BaseModel):
    model: str = "ollama/nomic-embed-text"


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
