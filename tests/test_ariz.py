"""Tests for ARIZ-85C deep analysis orchestrator."""

from unittest.mock import MagicMock, patch

import pytest

from triz_ai.engine.analyzer import AnalysisResult
from triz_ai.engine.ariz import (
    DeepAnalysisResult,
    PhysicalContradictionModel,
    ResourceInventory,
    SolutionVerification,
    StructuredProblemModel,
    SynthesizedSolution,
    TechnicalContradiction,
    VerifiedCandidate,
    _select_tools,
    orchestrate_deep,
)
from triz_ai.llm.client import TrizAIError


def _make_problem_model(recommended_tools=None):
    if recommended_tools is None:
        recommended_tools = ["technical_contradiction", "physical_contradiction", "su_field"]
    return StructuredProblemModel(
        original_problem="test problem",
        reformulated_problem="reformulated test problem",
        technical_contradiction_1=TechnicalContradiction(
            improving_param_id=1,
            improving_param_name="Weight of moving object",
            worsening_param_id=2,
            worsening_param_name="Weight of stationary object",
            intensified_description="TC1 intensified",
        ),
        technical_contradiction_2=TechnicalContradiction(
            improving_param_id=2,
            improving_param_name="Weight of stationary object",
            worsening_param_id=1,
            worsening_param_name="Weight of moving object",
            intensified_description="TC2 intensified",
        ),
        physical_contradiction=PhysicalContradictionModel(
            property="weight",
            macro_requirement="must be heavy",
            micro_requirement="must be light",
        ),
        ideal_final_result="The system ITSELF reduces weight without losing strength",
        resource_inventory=ResourceInventory(
            substances=["steel", "air"],
            fields=["mechanical"],
            time_resources=["idle time"],
            space_resources=["empty cavity"],
        ),
        recommended_tools=recommended_tools,
        reasoning="Test reasoning",
    )


def _make_verification(any_satisfies_ifr=True):
    return SolutionVerification(
        verified_candidates=[
            VerifiedCandidate(
                method="technical_contradiction",
                satisfies_ifr=True,
                ifr_gap="None",
                ideality_score=0.85,
                key_insight="Use composite material",
            ),
        ],
        any_satisfies_ifr=any_satisfies_ifr,
        synthesized_solutions=[
            SynthesizedSolution(
                title="Composite approach",
                description="Replace steel with carbon-fiber composite",
                principles_applied=["Principle 35: Parameter changes"],
                supersystem_changes=["New supplier for composite material"],
                ideality_score=0.9,
            ),
        ],
        reasoning="The composite approach satisfies the IFR.",
    )


class TestSelectTools:
    def test_always_includes_technical_contradiction(self):
        model = _make_problem_model(recommended_tools=["su_field", "trimming"])
        result = _select_tools(model)
        assert "technical_contradiction" in result

    def test_validates_against_known_methods(self):
        model = _make_problem_model(
            recommended_tools=["technical_contradiction", "bogus_method", "su_field"]
        )
        result = _select_tools(model)
        assert "bogus_method" not in result
        assert "technical_contradiction" in result
        assert "su_field" in result

    def test_clamps_to_max_4(self):
        model = _make_problem_model(
            recommended_tools=[
                "technical_contradiction",
                "physical_contradiction",
                "su_field",
                "function_analysis",
                "trimming",
                "trends",
            ]
        )
        result = _select_tools(model)
        assert len(result) == 4

    def test_minimum_2_tools(self):
        model = _make_problem_model(recommended_tools=["technical_contradiction"])
        result = _select_tools(model)
        assert len(result) >= 2

    def test_deduplicates(self):
        model = _make_problem_model(
            recommended_tools=["technical_contradiction", "technical_contradiction", "su_field"]
        )
        result = _select_tools(model)
        assert len(result) == len(set(result))


class TestOrchestrateDeep:
    @pytest.fixture
    def mock_llm(self):
        client = MagicMock()
        client.deep_reformulate.return_value = _make_problem_model()
        client.verify_and_synthesize.return_value = _make_verification(any_satisfies_ifr=True)
        return client

    @pytest.fixture
    def store(self, tmp_path):
        from triz_ai.patents.store import PatentStore

        db_path = tmp_path / "test.db"
        s = PatentStore(db_path=db_path)
        s.init_db()
        yield s
        s.close()

    def _patch_run_tools(self, return_value=None, side_effect=None):
        if return_value is None and side_effect is None:
            return_value = [AnalysisResult(problem="test", method="technical_contradiction")]
        return patch(
            "triz_ai.engine.ariz._run_tools",
            return_value=return_value,
            side_effect=side_effect,
        )

    def test_three_passes_called(self, mock_llm, store):
        with self._patch_run_tools() as mock_run:
            orchestrate_deep("test", mock_llm, store)
            mock_llm.deep_reformulate.assert_called_once()
            mock_run.assert_called_once()
            mock_llm.verify_and_synthesize.assert_called_once()

    def test_pass2_receives_reformulated_problem(self, mock_llm, store):
        with self._patch_run_tools() as mock_run:
            orchestrate_deep("test", mock_llm, store)
            call_args = mock_run.call_args
            # _run_tools(tools, problem_text, ifr, llm_client, store)
            problem_text = call_args[0][1]
            assert problem_text == "reformulated test problem"
            assert problem_text != "test"

    def test_pass2_receives_ifr(self, mock_llm, store):
        with self._patch_run_tools() as mock_run:
            orchestrate_deep("test", mock_llm, store)
            call_args = mock_run.call_args
            ifr = call_args[0][2]
            assert ifr == "The system ITSELF reduces weight without losing strength"

    def test_escape_hatch_triggers_on_no_ifr_satisfaction(self, mock_llm, store):
        mock_llm.verify_and_synthesize.side_effect = [
            _make_verification(any_satisfies_ifr=False),
            _make_verification(any_satisfies_ifr=True),
        ]
        tool_result = [AnalysisResult(problem="test", method="technical_contradiction")]
        with self._patch_run_tools(side_effect=[tool_result, tool_result]) as mock_run:
            result = orchestrate_deep("test", mock_llm, store)
            assert mock_run.call_count == 2
            assert mock_llm.verify_and_synthesize.call_count == 2
            assert result.used_escape_hatch is True

    def test_escape_hatch_max_one_retry(self, mock_llm, store):
        mock_llm.verify_and_synthesize.return_value = _make_verification(any_satisfies_ifr=False)
        tool_result = [AnalysisResult(problem="test", method="technical_contradiction")]
        with self._patch_run_tools(side_effect=[tool_result, tool_result]) as mock_run:
            result = orchestrate_deep("test", mock_llm, store)
            assert mock_run.call_count == 2
            assert result.used_escape_hatch is True

    def test_returns_deep_analysis_result(self, mock_llm, store):
        with self._patch_run_tools():
            result = orchestrate_deep("test", mock_llm, store)
            assert isinstance(result, DeepAnalysisResult)
            assert result.problem_model is not None
            assert len(result.tool_results) > 0
            assert len(result.tools_used) > 0
            assert result.verification is not None
            assert result.used_escape_hatch is False

    def test_all_pipelines_failing_raises(self, mock_llm, store):
        """If all pipelines fail, raise a clear error instead of passing empty candidates."""
        with (
            self._patch_run_tools(side_effect=TrizAIError("All TRIZ pipelines failed")),
            pytest.raises(TrizAIError, match="All TRIZ pipelines failed"),
        ):
            orchestrate_deep("test", mock_llm, store)

    def test_deep_model_forwarded_to_passes(self, mock_llm, store):
        """deep_model and reasoning_effort are forwarded to Passes 1 & 3."""
        with self._patch_run_tools():
            orchestrate_deep(
                "test",
                mock_llm,
                store,
                deep_model="reasoning-model",
                reasoning_effort="high",
            )
            # Pass 1
            mock_llm.deep_reformulate.assert_called_once_with(
                "test",
                model="reasoning-model",
                reasoning_effort="high",
                research_tool_descriptions=None,
            )
            # Pass 3
            call_kwargs = mock_llm.verify_and_synthesize.call_args[1]
            assert call_kwargs["model"] == "reasoning-model"
            assert call_kwargs["reasoning_effort"] == "high"

    def test_research_tools_passed_to_run_tools(self, mock_llm, store):
        """Research tools should be forwarded to _run_tools."""
        from triz_ai.tools import ResearchTool

        tool = ResearchTool(name="test", description="Test", fn=lambda q: [])
        with self._patch_run_tools() as mock_run:
            orchestrate_deep("test", mock_llm, store, research_tools=[tool])
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs.get("research_tools") == [tool]

    def test_llm_selects_research_tools(self, mock_llm, store):
        """LLM-recommended research tools should be filtered."""
        from triz_ai.tools import ResearchTool

        model = _make_problem_model()
        model.recommended_research_tools = ["arxiv"]
        mock_llm.deep_reformulate.return_value = model
        arxiv = ResearchTool(name="arxiv", description="Arxiv", fn=lambda q: [])
        web = ResearchTool(name="web", description="Web", fn=lambda q: [])
        with self._patch_run_tools() as mock_run:
            orchestrate_deep("test", mock_llm, store, research_tools=[arxiv, web])
            selected = mock_run.call_args[1].get("research_tools", [])
            assert len(selected) == 1
            assert selected[0].name == "arxiv"

    def test_no_recommendation_uses_all_tools(self, mock_llm, store):
        """Without LLM recommendation, all research tools should be used."""
        from triz_ai.tools import ResearchTool

        tool = ResearchTool(name="test", description="Test", fn=lambda q: [])
        with self._patch_run_tools() as mock_run:
            orchestrate_deep("test", mock_llm, store, research_tools=[tool])
            selected = mock_run.call_args[1].get("research_tools", [])
            assert len(selected) == 1

    def test_research_tool_descriptions_in_prompt(self, mock_llm, store):
        """Research tool descriptions should be passed to deep_reformulate."""
        from triz_ai.tools import ResearchTool

        tool = ResearchTool(name="bq", description="BigQuery patents", fn=lambda q: [])
        with self._patch_run_tools():
            orchestrate_deep("test", mock_llm, store, research_tools=[tool])
            call_kwargs = mock_llm.deep_reformulate.call_args[1]
            descs = call_kwargs.get("research_tool_descriptions")
            assert descs is not None
            assert len(descs) == 1
            assert descs[0]["name"] == "bq"
