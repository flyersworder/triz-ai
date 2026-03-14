"""Tests for configuration loading."""


def test_default_config():
    from triz_ai.config import Settings

    s = Settings()
    assert s.llm.default_model == "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
    assert s.embeddings.model == "openrouter/nvidia/llama-nemotron-embed-vl-1b-v2:free"
    assert s.embeddings.dimensions == 768
    assert s.evolution.review_threshold == 0.7


def test_default_database_path():
    from triz_ai.config import Settings

    s = Settings()
    assert s.database.path == "~/.triz-ai/patents.db"


def test_evolution_auto_classify_default():
    from triz_ai.config import Settings

    s = Settings()
    assert s.evolution.auto_classify is True
