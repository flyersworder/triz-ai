"""Tests for multi-tool TRIZ router."""

from unittest.mock import MagicMock, patch

import pytest

from triz_ai.engine.analyzer import AnalysisResult
from triz_ai.engine.router import _normalize_method, route
from triz_ai.llm.client import (
    IdealFinalResult,
    ProblemClassification,
    RootCauseAnalysis,
)


def _mock_result(method: str) -> AnalysisResult:
    return AnalysisResult(problem="test", method=method)


@pytest.fixture
def mock_llm():
    client = MagicMock()
    client.formulate_ifr.return_value = IdealFinalResult(
        ideal_result="The system ITSELF solves the problem without side effects.",
        reasoning="Ideal outcome",
    )
    client.classify_problem.return_value = ProblemClassification(
        primary_method="technical_contradiction",
        secondary_method="physical_contradiction",
        reasoning="Classic tradeoff problem",
        confidence=0.85,
        reformulated_problem="Increase X without worsening Y",
    )
    client.analyze_root_cause.return_value = RootCauseAnalysis(
        root_causes=["Surface cause", "Root cause"],
        reformulated_problem="Refined problem statement",
        reasoning="Traced via 5-whys",
    )
    client.get_embedding.return_value = [0.1] * 768
    return client


@pytest.fixture
def store(tmp_path):
    from triz_ai.patents.store import PatentStore

    db_path = tmp_path / "test.db"
    s = PatentStore(db_path=db_path)
    s.init_db()
    yield s
    s.close()


class TestNormalizeMethod:
    def test_underscore_form(self):
        assert _normalize_method("technical_contradiction") == "technical_contradiction"

    def test_hyphen_form(self):
        assert _normalize_method("technical-contradiction") == "technical_contradiction"

    def test_su_field_hyphen(self):
        assert _normalize_method("su-field") == "su_field"

    def test_case_insensitive(self):
        assert _normalize_method("SU-FIELD") == "su_field"

    def test_unknown_passes_through(self):
        assert _normalize_method("unknown") == "unknown"


class TestRouter:
    def _patch_pipeline(self, method_name):
        """Patch _get_pipeline to return a mock for the given method."""
        return patch(
            "triz_ai.engine.router._get_pipeline",
            return_value=MagicMock(return_value=_mock_result(method_name)),
        )

    def test_routes_to_technical_contradiction(self, mock_llm, store):
        with self._patch_pipeline("technical_contradiction") as mock_get:
            result = route("increase speed without weight", mock_llm, store)
            assert result.method == "technical_contradiction"
            mock_get.assert_called_once_with("technical_contradiction")

    def test_routes_to_physical_contradiction(self, mock_llm, store):
        mock_llm.classify_problem.return_value = ProblemClassification(
            primary_method="physical_contradiction",
            secondary_method=None,
            reasoning="Opposing requirements",
            confidence=0.9,
            reformulated_problem="Must be rigid and flexible",
        )
        with self._patch_pipeline("physical_contradiction") as mock_get:
            result = route("joint must be rigid AND flexible", mock_llm, store)
            assert result.method == "physical_contradiction"
            mock_get.assert_called_once_with("physical_contradiction")

    def test_routes_to_su_field(self, mock_llm, store):
        mock_llm.classify_problem.return_value = ProblemClassification(
            primary_method="su_field",
            secondary_method=None,
            reasoning="Detection problem",
            confidence=0.8,
            reformulated_problem="Detect cracks without sensors",
        )
        with self._patch_pipeline("su_field") as mock_get:
            result = route("detect delamination without sensors", mock_llm, store)
            assert result.method == "su_field"
            mock_get.assert_called_once_with("su_field")

    def test_routes_to_function_analysis(self, mock_llm, store):
        mock_llm.classify_problem.return_value = ProblemClassification(
            primary_method="function_analysis",
            secondary_method=None,
            reasoning="Harmful interaction",
            confidence=0.75,
            reformulated_problem="Adhesive damages die",
        )
        with self._patch_pipeline("function_analysis") as mock_get:
            result = route("adhesive damages the die", mock_llm, store)
            assert result.method == "function_analysis"
            mock_get.assert_called_once_with("function_analysis")

    def test_routes_to_trimming(self, mock_llm, store):
        mock_llm.classify_problem.return_value = ProblemClassification(
            primary_method="trimming",
            secondary_method=None,
            reasoning="Cost reduction",
            confidence=0.8,
            reformulated_problem="Reduce BOM cost",
        )
        with self._patch_pipeline("trimming") as mock_get:
            result = route("reduce BOM cost of gate driver", mock_llm, store)
            assert result.method == "trimming"
            mock_get.assert_called_once_with("trimming")

    def test_routes_to_trends(self, mock_llm, store):
        mock_llm.classify_problem.return_value = ProblemClassification(
            primary_method="trends",
            secondary_method=None,
            reasoning="Evolution question",
            confidence=0.85,
            reformulated_problem="Next gen packaging",
        )
        with self._patch_pipeline("trends") as mock_get:
            result = route("next generation SiC packaging", mock_llm, store)
            assert result.method == "trends"
            mock_get.assert_called_once_with("trends")

    def test_ifr_always_present(self, mock_llm, store):
        """IFR should be formulated for every analysis."""
        with self._patch_pipeline("technical_contradiction"):
            route("test problem", mock_llm, store)
            mock_llm.formulate_ifr.assert_called_once()

    def test_method_bypass_skips_classifier(self, mock_llm, store):
        """--method flag should bypass classifier."""
        with self._patch_pipeline("su_field") as mock_get:
            route("some problem", mock_llm, store, method="su-field")
            mock_llm.classify_problem.assert_not_called()
            mock_get.assert_called_once_with("su_field")

    def test_method_invalid_raises(self, mock_llm, store):
        with pytest.raises(ValueError, match="Unknown method"):
            route("test", mock_llm, store, method="nonexistent")

    def test_low_confidence_triggers_rca(self, mock_llm, store):
        """Low confidence classification should trigger root cause analysis."""
        mock_llm.classify_problem.return_value = ProblemClassification(
            primary_method="technical_contradiction",
            secondary_method=None,
            reasoning="Unclear",
            confidence=0.3,  # Below 0.4 threshold
            reformulated_problem="Vague problem",
        )
        with self._patch_pipeline("technical_contradiction"):
            route("vague problem", mock_llm, store)
            mock_llm.analyze_root_cause.assert_called_once()
            # classify_problem should be called twice (initial + after RCA)
            assert mock_llm.classify_problem.call_count == 2

    def test_secondary_method_attached(self, mock_llm, store):
        """Secondary method should be attached to result."""
        with self._patch_pipeline("technical_contradiction"):
            result = route("test", mock_llm, store)
            assert result.secondary_method == "physical_contradiction"

    def test_ifr_failure_continues(self, mock_llm, store):
        """IFR formulation failure should not block analysis."""
        mock_llm.formulate_ifr.side_effect = Exception("IFR failed")
        with self._patch_pipeline("technical_contradiction") as mock_get:
            route("test", mock_llm, store)
            mock_get.assert_called_once()
            # Pipeline should receive None as ideal_final_result
            pipeline_fn = mock_get.return_value
            call_args = pipeline_fn.call_args
            assert call_args[0][1] is None  # ideal_final_result

    def test_rca_failure_uses_original_classification(self, mock_llm, store):
        """RCA failure should fall back to original low-confidence classification."""
        mock_llm.classify_problem.return_value = ProblemClassification(
            primary_method="technical_contradiction",
            secondary_method=None,
            reasoning="Unclear",
            confidence=0.3,
            reformulated_problem="Vague",
        )
        mock_llm.analyze_root_cause.side_effect = Exception("RCA failed")
        with self._patch_pipeline("technical_contradiction"):
            route("vague problem", mock_llm, store)
            # RCA was attempted
            mock_llm.analyze_root_cause.assert_called_once()
            # But classify was only called once (no re-classify after failed RCA)
            assert mock_llm.classify_problem.call_count == 1

    def test_research_tools_passed_to_pipeline(self, mock_llm, store):
        """Research tools should be forwarded to the pipeline."""
        from triz_ai.tools import ResearchTool

        tool = ResearchTool(name="test", description="test", fn=lambda q, ctx: [])
        with self._patch_pipeline("technical_contradiction") as mock_get:
            route("test", mock_llm, store, research_tools=[tool])
            pipeline_fn = mock_get.return_value
            call_kwargs = pipeline_fn.call_args[1]
            assert call_kwargs.get("research_tools") == [tool]

    def test_context_tools_enrich_problem_text(self, mock_llm, store):
        """Context-stage tools should enrich problem_text before pipeline dispatch."""
        from triz_ai.tools import ResearchTool

        tool = ResearchTool(
            name="web",
            description="Web search",
            fn=lambda q, ctx: [{"content": "SiC MOSFETs switch at 100-400kHz"}],
            stages=["context"],
        )
        with self._patch_pipeline("technical_contradiction") as mock_get:
            route("SiC problem", mock_llm, store, research_tools=[tool])
            pipeline_fn = mock_get.return_value
            call_args = pipeline_fn.call_args[0]
            # problem_text (first arg) should contain the context
            assert "100-400kHz" in call_args[0]

    def test_context_tools_run_before_ifr(self, mock_llm, store):
        """Context tools should enrich problem_text before IFR formulation."""
        from triz_ai.tools import ResearchTool

        tool = ResearchTool(
            name="web",
            description="Web search",
            fn=lambda q, ctx: [{"content": "Extra domain context"}],
            stages=["context"],
        )
        with self._patch_pipeline("technical_contradiction"):
            route("test problem", mock_llm, store, research_tools=[tool])
            # IFR should receive the enriched problem text
            ifr_call_args = mock_llm.formulate_ifr.call_args[0]
            assert "Extra domain context" in ifr_call_args[0]

    def test_context_tools_no_content_no_change(self, mock_llm, store):
        """Context tools returning empty content should not modify problem text."""
        from triz_ai.tools import ResearchTool

        tool = ResearchTool(
            name="empty",
            description="Returns nothing useful",
            fn=lambda q, ctx: [{"content": ""}],
            stages=["context"],
        )
        with self._patch_pipeline("technical_contradiction") as mock_get:
            route("original problem", mock_llm, store, research_tools=[tool])
            pipeline_fn = mock_get.return_value
            call_args = pipeline_fn.call_args[0]
            assert call_args[0] == "original problem"


class TestSelfEvolutionHook:
    """Self-evolution collection hooks in route()."""

    def test_collects_web_results_after_analysis(self, mock_llm, store):
        """route() should call collect_search_observations when research_tools present."""
        from triz_ai.tools import ResearchTool

        def mock_search(query, context):
            return [
                {"title": "Web Result 1", "abstract": "Found via search", "source": "test_tool"},
            ]

        tool = ResearchTool(
            name="test_tool",
            description="Test search",
            fn=mock_search,
            stages=["search"],
        )

        mock_result = AnalysisResult(
            problem="test problem",
            method="technical_contradiction",
            improving_param={"id": 1, "name": "Weight"},
            worsening_param={"id": 2, "name": "Length"},
            recommended_principles=[{"id": 10, "name": "Prior action", "description": "..."}],
            patent_examples=[
                {"title": "Web Result 1", "abstract": "Found via search", "source": "test_tool"},
            ],
        )

        with patch(
            "triz_ai.engine.router._get_pipeline",
            return_value=MagicMock(return_value=mock_result),
        ):
            route("test problem", mock_llm, store, research_tools=[tool])

        observations = store.get_unconsolidated_observations()
        assert len(observations) == 1
        assert observations[0].source_tool == "test_tool"

    def test_no_collection_without_research_tools(self, mock_llm, store):
        """route() without research_tools should not collect anything."""
        with patch(
            "triz_ai.engine.router._get_pipeline",
            return_value=MagicMock(return_value=AnalysisResult(problem="test")),
        ):
            route("test", mock_llm, store)

        observations = store.get_unconsolidated_observations()
        assert len(observations) == 0

    def test_collection_failure_does_not_break_analysis(self, mock_llm, store):
        """If collection fails, route() should still return the analysis result."""
        from triz_ai.tools import ResearchTool

        tool = ResearchTool(name="t", description="t", fn=lambda q, c: [], stages=["search"])
        mock_result = AnalysisResult(problem="test", method="technical_contradiction")

        with (
            patch(
                "triz_ai.engine.router._get_pipeline",
                return_value=MagicMock(return_value=mock_result),
            ),
            patch(
                "triz_ai.evolution.self_evolve.collect_search_observations",
                side_effect=Exception("DB error"),
            ),
        ):
            result = route("test", mock_llm, store, research_tools=[tool])
            assert result.problem == "test"
