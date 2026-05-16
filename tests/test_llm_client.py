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
    _strictify_schema,
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

    def test_complete_openai_fallback_retries_on_malformed_response(self):
        """The retry-with-stricter-prompt fallback must also work on the
        openai-SDK path (the litellm path is covered by
        TestComplete.test_retries_on_malformed_response). This guards against
        accidentally branching the retry logic to only one backend.
        """
        invalid_json = '{"bad": "data"}'
        valid_json = json.dumps(
            {
                "improving_param": 5,
                "worsening_param": 10,
                "reasoning": "retried successfully on openai path",
            }
        )

        def make_response(content):
            message = MagicMock()
            message.content = content
            choice = MagicMock()
            choice.message = message
            r = MagicMock()
            r.choices = [choice]
            return r

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            make_response(invalid_json),
            make_response(valid_json),
        ]

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
            assert result.improving_param == 5
            assert mock_client.chat.completions.create.call_count == 2

    def test_complete_openai_fallback_uses_json_schema_strict(self):
        """openai-SDK fallback must also send response_format=json_schema strict.

        Issue #18 fix: previously this path also used json_object loose mode.
        Both backends share the same upgrade.
        """
        valid_json = json.dumps(
            {
                "improving_param": 1,
                "worsening_param": 2,
                "reasoning": "strict openai path",
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
            client._complete("system", "user", ExtractedContradiction)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        rf = call_kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "ExtractedContradiction"
        assert rf["json_schema"]["strict"] is True
        schema = rf["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"].keys())

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


class TestCompleteResponseFormat:
    """Issue #18 fix: _complete must request strict json_schema, not loose json_object.

    The behavior change is in the `response_format` argument passed to the LLM.
    Loose json_object lets the model emit malformed JSON (reproducibly on
    nemotron-3-super-120b-a12b:free with deep-mode schemas — see CLAUDE.md
    pre-0.18.0 note); strict json_schema enforces the pydantic shape
    server-side and across providers via LiteLLM's translation layer.
    """

    def test_litellm_path_sends_json_schema_strict(self, client):
        """litellm.completion must receive response_format.type=json_schema strict."""
        valid_json = json.dumps(
            {
                "improving_param": 1,
                "worsening_param": 2,
                "reasoning": "strict litellm path",
            }
        )
        with patch(
            "litellm.completion",
            return_value=_make_completion_response(valid_json),
        ) as mock_completion:
            client._complete("system", "user", ExtractedContradiction)

        kwargs = mock_completion.call_args.kwargs
        rf = kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "ExtractedContradiction"
        assert rf["json_schema"]["strict"] is True
        # Schema is strictified inline — additionalProperties=false and full required.
        schema = rf["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"].keys())

    def test_litellm_path_strictifies_nested_pydantic_models(self, client):
        """Nested models (StructuredProblemModel) must be strictified end-to-end.

        Deep schemas with $defs / $ref / Optional fields with default=None are
        the case the bug was about. The strictifier must reach into $defs too.
        """
        from triz_ai.engine.ariz import (
            ResourceInventory,
            StructuredProblemModel,
            TechnicalContradiction,
        )

        sample = StructuredProblemModel(
            original_problem="o",
            reformulated_problem="r",
            technical_contradiction_1=TechnicalContradiction(
                improving_param_id=1,
                improving_param_name="x",
                worsening_param_id=2,
                worsening_param_name="y",
                intensified_description="z",
            ),
            technical_contradiction_2=TechnicalContradiction(
                improving_param_id=2,
                improving_param_name="y",
                worsening_param_id=1,
                worsening_param_name="x",
                intensified_description="z",
            ),
            ideal_final_result="ifr",
            resource_inventory=ResourceInventory(
                substances=[],
                fields=[],
                time_resources=[],
                space_resources=[],
            ),
            recommended_tools=["technical_contradiction"],
            reasoning="r",
        )
        with patch(
            "litellm.completion",
            return_value=_make_completion_response(sample.model_dump_json()),
        ) as mock_completion:
            client._complete("system", "user", StructuredProblemModel)

        schema = mock_completion.call_args.kwargs["response_format"]["json_schema"]["schema"]
        # $defs nested objects must each have additionalProperties=false too
        for def_name, def_schema in schema.get("$defs", {}).items():
            assert def_schema.get("additionalProperties") is False, (
                f"$defs.{def_name} missing additionalProperties=false"
            )
            assert set(def_schema.get("required", [])) == set(
                def_schema.get("properties", {}).keys()
            ), f"$defs.{def_name} required != properties"
        # Optional field (physical_contradiction) must appear in required
        # (strict mode forbids omission — nullability is via anyOf+null, not absence)
        assert "physical_contradiction" in schema["required"]

        # `default` keys must be stripped everywhere
        def _no_defaults(node):
            if isinstance(node, dict):
                assert "default" not in node, f"default key still present: {node!r}"
                for v in node.values():
                    _no_defaults(v)
            elif isinstance(node, list):
                for v in node:
                    _no_defaults(v)

        _no_defaults(schema)


class TestStrictifySchema:
    """Unit tests for _strictify_schema — covers edge cases the integration
    tests touch but don't isolate (e.g. nested objects without `type: object`,
    deeply-nested $defs, idempotency)."""

    def test_adds_additional_properties_false_on_object(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        out = _strictify_schema(schema)
        assert out["additionalProperties"] is False

    def test_required_covers_all_properties(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a"],  # incomplete required — must be expanded
        }
        out = _strictify_schema(schema)
        assert set(out["required"]) == {"a", "b"}

    def test_strips_default_keys_everywhere(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string", "default": "hello"},
                "b": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "default": None,
                },
            },
        }
        out = _strictify_schema(schema)
        assert "default" not in out["properties"]["a"]
        assert "default" not in out["properties"]["b"]

    def test_preserves_anyof_null_unions(self):
        """Optional pydantic fields (`X | None = None`) become anyOf+null.
        Strict mode handles this via type-union nullability — must NOT be
        rewritten to a non-nullable shape."""
        schema = {
            "type": "object",
            "properties": {
                "maybe": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                }
            },
        }
        out = _strictify_schema(schema)
        # anyOf shape preserved
        assert out["properties"]["maybe"]["anyOf"] == [
            {"type": "string"},
            {"type": "null"},
        ]
        # default stripped, field is required
        assert "default" not in out["properties"]["maybe"]
        assert "maybe" in out["required"]

    def test_recurses_into_defs(self):
        schema = {
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"x": {"type": "integer", "default": 0}},
                }
            },
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/Inner"}},
        }
        out = _strictify_schema(schema)
        inner = out["$defs"]["Inner"]
        assert inner["additionalProperties"] is False
        assert inner["required"] == ["x"]
        assert "default" not in inner["properties"]["x"]

    def test_recurses_into_array_items(self):
        schema = {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer", "default": 0}},
                    },
                }
            },
        }
        out = _strictify_schema(schema)
        item = out["properties"]["rows"]["items"]
        assert item["additionalProperties"] is False
        assert item["required"] == ["id"]
        assert "default" not in item["properties"]["id"]

    def test_does_not_mutate_input(self):
        """Strictifier must return a defensive copy — pydantic callers may
        reuse the schema dict, and mutation would surprise them."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string", "default": "x"}},
        }
        original = json.loads(json.dumps(schema))  # deep snapshot
        _strictify_schema(schema)
        assert schema == original, "input schema was mutated"

    def test_idempotent(self):
        """Applying the strictifier twice should yield the same result."""
        from triz_ai.llm.client import ProblemClassification

        once = _strictify_schema(ProblemClassification.model_json_schema())
        twice = _strictify_schema(once)
        assert once == twice

    def test_walks_into_oneof_and_allof_branches(self):
        """Strictifier walks into oneOf/allOf branches and applies object rules.

        The strictifier doesn't rewrite the oneOf/allOf keywords themselves —
        OpenAI strict mode has caveats around them — but each branch's object
        rules (additionalProperties=false, required = all properties, no
        default keys) must still be applied so that if a future schema does
        use these keywords, branch-level shape is correct.
        """
        schema = {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {"a": {"type": "string", "default": "x"}},
                },
                {"type": "object", "properties": {"b": {"type": "integer"}}},
            ],
            "allOf": [
                {
                    "type": "object",
                    "properties": {"c": {"type": "boolean", "default": True}},
                },
            ],
        }
        out = _strictify_schema(schema)
        # oneOf keyword preserved, but branches strictified
        for branch in out["oneOf"]:
            assert branch["additionalProperties"] is False
            assert set(branch["required"]) == set(branch["properties"].keys())
            for prop_schema in branch["properties"].values():
                assert "default" not in prop_schema
        # allOf same
        for branch in out["allOf"]:
            assert branch["additionalProperties"] is False
            assert set(branch["required"]) == set(branch["properties"].keys())

    def test_no_oneof_or_allof_in_response_models(self):
        """Canary: none of the response models passed to _complete should
        emit oneOf/allOf today. If pydantic starts emitting them (e.g. someone
        adds a discriminated union), the strictifier's docstring caveat needs
        revisiting — branches get walked but the keywords themselves pass
        through, and OpenAI strict mode does not fully support them.
        """
        from triz_ai.engine.ariz import SolutionVerification, StructuredProblemModel
        from triz_ai.llm.client import (
            CandidateParameterProposal,
            CandidatePrincipleProposal,
            ExtractedContradiction,
            FunctionAnalysisResult,
            IdeaBatch,
            IdealFinalResult,
            MatrixSeedResult,
            ObservationValidationBatch,
            PatentClassification,
            PhysicalContradictionResult,
            ProblemClassification,
            RootCauseAnalysis,
            SolutionDirectionBatch,
            SuFieldResult,
            TrendsResult,
            TrimmingResult,
        )

        response_models = [
            ExtractedContradiction,
            PatentClassification,
            IdeaBatch,
            SolutionDirectionBatch,
            CandidatePrincipleProposal,
            CandidateParameterProposal,
            MatrixSeedResult,
            ObservationValidationBatch,
            ProblemClassification,
            IdealFinalResult,
            RootCauseAnalysis,
            PhysicalContradictionResult,
            SuFieldResult,
            FunctionAnalysisResult,
            TrimmingResult,
            TrendsResult,
            StructuredProblemModel,
            SolutionVerification,
        ]

        def _contains_keyword(node, keyword):
            if isinstance(node, dict):
                if keyword in node:
                    return True
                return any(_contains_keyword(v, keyword) for v in node.values())
            if isinstance(node, list):
                return any(_contains_keyword(v, keyword) for v in node)
            return False

        for model in response_models:
            schema = model.model_json_schema()
            assert not _contains_keyword(schema, "oneOf"), (
                f"{model.__name__} emits oneOf — strictifier docstring caveat applies; "
                "verify OpenAI strict mode accepts it on your provider before shipping."
            )
            assert not _contains_keyword(schema, "allOf"), (
                f"{model.__name__} emits allOf — strictifier docstring caveat applies."
            )
