"""Tests for configuration loading."""


def test_default_config():
    from triz_ai.config import Settings

    s = Settings()
    assert s.llm.default_model == "openrouter/google/gemini-2.5-flash"
    assert s.embeddings.model == "ollama/nomic-embed-text"
    assert s.evolution.review_threshold == 0.7


def test_default_database_path():
    from triz_ai.config import Settings

    s = Settings()
    assert s.database.path == "~/.triz-ai/patents.db"


def test_evolution_auto_classify_default():
    from triz_ai.config import Settings

    s = Settings()
    assert s.evolution.auto_classify is True
