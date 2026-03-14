"""Tests for LLM client with mocked litellm calls."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from triz_ai.llm.client import (
    ExtractedContradiction,
    LLMClient,
    PatentClassification,
)


def _make_completion_response(content: str):
    """Create a mock litellm completion response."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _make_embedding_response(embedding: list[float]):
    """Create a mock litellm embedding response."""
    return SimpleNamespace(data=[{"embedding": embedding}])


@pytest.fixture
def client():
    with patch("triz_ai.llm.client.load_config") as mock_config:
        mock_settings = MagicMock()
        mock_settings.llm.default_model = "test-model"
        mock_settings.embeddings.model = "test-embed-model"
        mock_config.return_value = mock_settings
        yield LLMClient()


class TestComplete:
    """Tests for the _complete method."""

    def test_validates_pydantic_model(self, client):
        valid_json = json.dumps(
            {
                "improving_param": 1,
                "worsening_param": 2,
                "reasoning": "test reasoning",
            }
        )
        with patch("litellm.completion", return_value=_make_completion_response(valid_json)):
            result = client._complete("system", "user", ExtractedContradiction)
            assert isinstance(result, ExtractedContradiction)
            assert result.improving_param == 1
            assert result.worsening_param == 2
            assert result.reasoning == "test reasoning"

    def test_retries_on_malformed_response(self, client):
        invalid_json = '{"bad": "data"}'
        valid_json = json.dumps(
            {
                "improving_param": 5,
                "worsening_param": 10,
                "reasoning": "retried successfully",
            }
        )
        responses = [
            _make_completion_response(invalid_json),
            _make_completion_response(valid_json),
        ]
        with patch("litellm.completion", side_effect=responses):
            result = client._complete("system", "user", ExtractedContradiction)
            assert isinstance(result, ExtractedContradiction)
            assert result.improving_param == 5

    def test_raises_on_second_failure(self, client):
        invalid_json = '{"bad": "data"}'
        responses = [
            _make_completion_response(invalid_json),
            _make_completion_response(invalid_json),
        ]
        with (
            patch("litellm.completion", side_effect=responses),
            pytest.raises(  # noqa: B017
                Exception
            ),
        ):
            client._complete("system", "user", ExtractedContradiction)


class TestExtractContradiction:
    """Tests for extract_contradiction."""

    def test_returns_extracted_contradiction(self, client):
        valid_json = json.dumps(
            {
                "improving_param": 14,
                "worsening_param": 26,
                "reasoning": "Strength vs complexity",
            }
        )
        with patch("litellm.completion", return_value=_make_completion_response(valid_json)):
            result = client.extract_contradiction("Make it stronger without more weight")
            assert isinstance(result, ExtractedContradiction)
            assert result.improving_param == 14
            assert result.worsening_param == 26


class TestClassifyPatent:
    """Tests for classify_patent."""

    def test_returns_patent_classification(self, client):
        valid_json = json.dumps(
            {
                "principle_ids": [1, 35],
                "contradiction": {"improving": 14, "worsening": 26},
                "confidence": 0.85,
                "reasoning": "Uses segmentation and parameter changes",
            }
        )
        with patch("litellm.completion", return_value=_make_completion_response(valid_json)):
            result = client.classify_patent("Patent about segmented structures...")
            assert isinstance(result, PatentClassification)
            assert result.principle_ids == [1, 35]
            assert result.confidence == 0.85


class TestGetEmbedding:
    """Tests for get_embedding."""

    def test_returns_list_of_floats(self, client):
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        with patch("litellm.embedding", return_value=_make_embedding_response(embedding)):
            result = client.get_embedding("test text")
            assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
            assert all(isinstance(x, float) for x in result)
