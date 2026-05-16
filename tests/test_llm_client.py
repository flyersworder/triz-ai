"""Tests for LLM client with mocked litellm and openai fallback calls."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from triz_ai.llm.client import (
    ExtractedContradiction,
    LLMClient,
    PatentClassification,
    TrizAIError,
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


class TestValidateObservations:
    """Tests for validate_observations."""

    def test_validate_observations_returns_validated_results(self, client):
        """validate_observations should return a list of validated principle assignments."""
        from triz_ai.llm.client import (
            ObservationValidation,
            ObservationValidationBatch,
            ValidatedPrinciple,
        )

        client._complete = MagicMock(
            return_value=ObservationValidationBatch(
                validations=[
                    ObservationValidation(
                        observation_id="ws:abc123",
                        validated_principles=[
                            ValidatedPrinciple(principle_id=35, confidence=0.8),
                            ValidatedPrinciple(principle_id=2, confidence=0.3),
                        ],
                    ),
                ]
            )
        )

        results = client.validate_observations(
            observations=[
                {
                    "id": "ws:abc123",
                    "title": "PCM Thermal Management",
                    "snippet": "Phase change materials for cooling",
                }
            ],
            improving_param=17,
            improving_name="Temperature",
            worsening_param=14,
            worsening_name="Strength",
            principle_ids=[35, 2],
        )

        assert len(results.validations) == 1
        assert results.validations[0].observation_id == "ws:abc123"
        assert len(results.validations[0].validated_principles) == 2


class TestVerifyAndSynthesizePayload:
    """Tests for the verify_and_synthesize user-prompt payload shape."""

    def test_user_prompt_includes_direction_descriptions_and_principles(self, client):
        """Pass 3 must see full direction context, not just titles — without
        descriptions and principles_applied it cannot cluster across methods."""
        from triz_ai.engine.ariz import (
            PhysicalContradictionModel,
            ResourceInventory,
            SolutionVerification,
            StructuredProblemModel,
            TechnicalContradiction,
        )

        captured: dict = {}

        def fake_complete(system_prompt, user_prompt, response_model, **kwargs):
            captured["system"] = system_prompt
            captured["user"] = user_prompt
            return SolutionVerification(
                verified_candidates=[],
                any_satisfies_ifr=False,
                synthesized_solutions=[],
                reasoning="stub",
            )

        client._complete = fake_complete

        problem_model = StructuredProblemModel(
            original_problem="orig",
            reformulated_problem="reformulated",
            technical_contradiction_1=TechnicalContradiction(
                improving_param_id=9,
                improving_param_name="Speed",
                worsening_param_id=31,
                worsening_param_name="Harmful side effects",
                intensified_description="Faster switching → more EMI",
            ),
            technical_contradiction_2=TechnicalContradiction(
                improving_param_id=31,
                improving_param_name="Harmful side effects",
                worsening_param_id=9,
                worsening_param_name="Speed",
                intensified_description="Less EMI → slower switching",
            ),
            physical_contradiction=PhysicalContradictionModel(
                property="dv/dt",
                macro_requirement="fast",
                micro_requirement="slow",
            ),
            ideal_final_result="The system itself switches fast without EMI",
            resource_inventory=ResourceInventory(
                substances=[], fields=[], time_resources=[], space_resources=[]
            ),
            recommended_tools=["technical_contradiction"],
            reasoning="r",
        )

        candidates = [
            {
                "method": "technical_contradiction",
                "reasoning": "TC reasoning",
                "solution_directions": [
                    {
                        "title": "Closed-loop dv/dt control",
                        "description": "Sense phase-node ringing and adjust gate drive",
                        "principles_applied": ["Principle 15: Dynamics"],
                    },
                ],
            },
            {
                "method": "su_field",
                "reasoning": "Su-Field reasoning",
                "solution_directions": [
                    {
                        "title": "Two-level gate drive",
                        "description": "Fast-through-Miller, slow-at-excitation window",
                        "principles_applied": ["Principle 15: Dynamics"],
                    },
                ],
            },
        ]

        client.verify_and_synthesize(problem_model, candidates)

        user = captured["user"]
        # Descriptions, not just titles, must reach Pass 3
        assert "Sense phase-node ringing and adjust gate drive" in user
        assert "Fast-through-Miller, slow-at-excitation window" in user
        # Principles must reach Pass 3 — they are the strongest cross-method
        # clustering hint
        assert "Principle 15: Dynamics" in user
        # Method labels still present
        assert "technical_contradiction" in user
        assert "su_field" in user

    def test_clamps_hallucinated_source_titles_and_methods(self, client):
        """LLM may paraphrase titles or invent methods despite the prompt's
        'exact match' instruction. The post-call clamp must drop entries that
        aren't present in the input candidates — otherwise the CLI prints
        fabricated 'Merged from:' provenance and inflated concordance badges."""
        from triz_ai.engine.ariz import (
            ResourceInventory,
            SolutionVerification,
            StructuredProblemModel,
            SynthesizedSolution,
            TechnicalContradiction,
        )

        # LLM returns a mix of valid + hallucinated titles and methods
        hallucinated_response = SolutionVerification(
            verified_candidates=[],
            any_satisfies_ifr=True,
            synthesized_solutions=[
                SynthesizedSolution(
                    title="Adaptive gate-slew",
                    description="merged",
                    principles_applied=["Principle 15: Dynamics"],
                    supersystem_changes=[],
                    ideality_score=0.8,
                    supported_by_methods=[
                        "technical_contradiction",  # valid
                        "su_field",  # valid
                        "imagined_method",  # hallucinated — not in candidates
                    ],
                    source_direction_titles=[
                        "Closed-loop dv/dt control",  # exact match
                        "Adaptive dv/dt closed-loop control",  # paraphrase
                        "Two-level gate drive",  # exact match
                    ],
                ),
            ],
            reasoning="r",
        )

        client._complete = lambda *a, **kw: hallucinated_response

        problem_model = StructuredProblemModel(
            original_problem="o",
            reformulated_problem="r",
            technical_contradiction_1=TechnicalContradiction(
                improving_param_id=9,
                improving_param_name="Speed",
                worsening_param_id=31,
                worsening_param_name="Harmful side effects",
                intensified_description="x",
            ),
            technical_contradiction_2=TechnicalContradiction(
                improving_param_id=31,
                improving_param_name="Harmful side effects",
                worsening_param_id=9,
                worsening_param_name="Speed",
                intensified_description="y",
            ),
            ideal_final_result="ifr",
            resource_inventory=ResourceInventory(
                substances=[], fields=[], time_resources=[], space_resources=[]
            ),
            recommended_tools=["technical_contradiction"],
            reasoning="r",
        )

        candidates = [
            {
                "method": "technical_contradiction",
                "reasoning": "tc",
                "solution_directions": [
                    {
                        "title": "Closed-loop dv/dt control",
                        "description": "d",
                        "principles_applied": ["Principle 15: Dynamics"],
                    }
                ],
            },
            {
                "method": "su_field",
                "reasoning": "sf",
                "solution_directions": [
                    {
                        "title": "Two-level gate drive",
                        "description": "d",
                        "principles_applied": ["Principle 15: Dynamics"],
                    }
                ],
            },
        ]

        result = client.verify_and_synthesize(problem_model, candidates)
        sol = result.synthesized_solutions[0]

        # Paraphrased title is dropped; exact matches preserved
        assert sol.source_direction_titles == [
            "Closed-loop dv/dt control",
            "Two-level gate drive",
        ]
        # Hallucinated method is dropped; valid methods preserved in order
        assert sol.supported_by_methods == ["technical_contradiction", "su_field"]


class TestGetEmbedding:
    """Tests for get_embedding."""

    def test_returns_list_of_floats(self, client):
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        with patch("litellm.embedding", return_value=_make_embedding_response(embedding)):
            result = client.get_embedding("test text")
            assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
            assert all(isinstance(x, float) for x in result)


class TestOpenAIFallback:
    """Tests for the openai SDK fallback when litellm is not installed."""

    def test_complete_uses_openai_when_no_litellm(self):
        """When HAS_LITELLM is False and api_base is set, use openai SDK."""
        valid_json = json.dumps(
            {
                "improving_param": 1,
                "worsening_param": 2,
                "reasoning": "openai fallback test",
            }
        )
        message = MagicMock()
        message.content = valid_json
        choice = MagicMock()
        choice.message = message
        mock_response = MagicMock()
        mock_response.choices = [choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with (
            patch("triz_ai.llm.client.load_config") as mock_config,
            patch("triz_ai.llm.client.HAS_LITELLM", False),
            patch("triz_ai.llm.client.openai.OpenAI", return_value=mock_client),
        ):
            mock_settings = MagicMock()
            mock_settings.llm.default_model = "test-model"
            mock_settings.llm.api_base = "http://localhost:4000"
            mock_settings.llm.api_key = "test-key"
            mock_settings.llm.ssl_verify = True
            mock_settings.embeddings.model = "test-embed-model"
            mock_config.return_value = mock_settings

            client = LLMClient()
            result = client._complete("system", "user", ExtractedContradiction)
            assert isinstance(result, ExtractedContradiction)
            assert result.improving_param == 1
            mock_client.chat.completions.create.assert_called_once()

    def test_embedding_uses_openai_when_no_litellm(self):
        """When HAS_LITELLM is False and api_base is set, use openai SDK for embeddings."""
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with (
            patch("triz_ai.llm.client.load_config") as mock_config,
            patch("triz_ai.llm.client.HAS_LITELLM", False),
            patch("triz_ai.llm.client.openai.OpenAI", return_value=mock_client),
        ):
            mock_settings = MagicMock()
            mock_settings.llm.default_model = "test-model"
            mock_settings.embeddings.model = "test-embed-model"
            mock_settings.embeddings.api_base = "http://localhost:4000"
            mock_settings.embeddings.api_key = "test-key"
            mock_settings.llm.api_base = "http://localhost:4000"
            mock_settings.llm.ssl_verify = True
            mock_config.return_value = mock_settings

            client = LLMClient()
            result = client.get_embedding("test text")
            assert result == [0.1, 0.2, 0.3]
            mock_client.embeddings.create.assert_called_once()

    def test_raises_when_no_litellm_and_no_api_base(self):
        """When HAS_LITELLM is False and no api_base, raise TrizAIError."""
        with (
            patch("triz_ai.llm.client.load_config") as mock_config,
            patch("triz_ai.llm.client.HAS_LITELLM", False),
        ):
            mock_settings = MagicMock()
            mock_settings.llm.default_model = "test-model"
            mock_settings.llm.api_base = None
            mock_settings.llm.api_key = None
            mock_settings.llm.ssl_verify = True
            mock_settings.embeddings.model = "test-embed-model"
            mock_settings.embeddings.api_base = None
            mock_settings.embeddings.api_key = None
            mock_config.return_value = mock_settings

            client = LLMClient()
            with pytest.raises(TrizAIError, match="No LLM backend available"):
                client._complete("system", "user", ExtractedContradiction)

    def test_raises_when_no_litellm_and_no_embedding_api_base(self):
        """When HAS_LITELLM is False and no embedding_api_base, raise TrizAIError."""
        with (
            patch("triz_ai.llm.client.load_config") as mock_config,
            patch("triz_ai.llm.client.HAS_LITELLM", False),
        ):
            mock_settings = MagicMock()
            mock_settings.llm.default_model = "test-model"
            mock_settings.llm.api_base = "http://localhost:4000"
            mock_settings.llm.api_key = "test-key"
            mock_settings.llm.ssl_verify = True
            mock_settings.embeddings.model = "test-embed-model"
            mock_settings.embeddings.api_base = None
            mock_settings.embeddings.api_key = None
            mock_config.return_value = mock_settings

            client = LLMClient()
            with pytest.raises(TrizAIError, match="embeddings.api_base"):
                client.get_embedding("test text")
