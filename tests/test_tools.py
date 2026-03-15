"""Tests for pluggable research tools interface."""

from unittest.mock import MagicMock

from triz_ai.engine.analyzer import run_enrichment_tools, search_patents
from triz_ai.tools import ResearchTool, run_stage_tools


def _mock_llm():
    client = MagicMock()
    client.get_embedding.return_value = [0.1] * 768
    return client


class TestResearchToolDataclass:
    def test_fields(self):
        tool = ResearchTool(name="test", description="A test tool", fn=lambda q, ctx: [])
        assert tool.name == "test"
        assert tool.description == "A test tool"
        assert callable(tool.fn)

    def test_default_stages(self):
        tool = ResearchTool(name="test", description="Test", fn=lambda q, ctx: [])
        assert tool.stages == ["search"]

    def test_custom_stages(self):
        tool = ResearchTool(
            name="test",
            description="Test",
            fn=lambda q, ctx: [],
            stages=["context", "enrichment"],
        )
        assert tool.stages == ["context", "enrichment"]


class TestRunStageTools:
    def test_filters_by_stage(self):
        search_tool = ResearchTool(
            name="search",
            description="Search",
            fn=lambda q, ctx: [{"title": "A"}],
            stages=["search"],
        )
        context_tool = ResearchTool(
            name="context",
            description="Context",
            fn=lambda q, ctx: [{"content": "B"}],
            stages=["context"],
        )
        results = run_stage_tools([search_tool, context_tool], "search", "query")
        assert len(results) == 1
        assert results[0]["title"] == "A"

    def test_passes_context(self):
        captured = {}

        def capture_fn(q, ctx):
            captured.update(ctx)
            return []

        tool = ResearchTool(name="t", description="T", fn=capture_fn, stages=["search"])
        run_stage_tools([tool], "search", "query", extra_context={"principle_ids": [1, 2]})
        assert captured["stage"] == "search"
        assert captured["principle_ids"] == [1, 2]

    def test_empty_tools_returns_empty(self):
        assert run_stage_tools(None, "search", "query") == []
        assert run_stage_tools([], "search", "query") == []

    def test_failing_tool_skipped(self):
        def failing_fn(q, ctx):
            raise RuntimeError("fail")

        good_tool = ResearchTool(
            name="good",
            description="Good",
            fn=lambda q, ctx: [{"content": "ok"}],
            stages=["context"],
        )
        bad_tool = ResearchTool(name="bad", description="Bad", fn=failing_fn, stages=["context"])
        results = run_stage_tools([bad_tool, good_tool], "context", "query")
        assert len(results) == 1
        assert results[0]["content"] == "ok"

    def test_multi_stage_tool(self):
        tool = ResearchTool(
            name="multi",
            description="Multi",
            fn=lambda q, ctx: [{"content": f"stage={ctx['stage']}"}],
            stages=["context", "enrichment"],
        )
        ctx_results = run_stage_tools([tool], "context", "query")
        assert ctx_results[0]["content"] == "stage=context"
        enrich_results = run_stage_tools([tool], "enrichment", "query")
        assert enrich_results[0]["content"] == "stage=enrichment"
        # Should not run at search stage
        search_results = run_stage_tools([tool], "search", "query")
        assert search_results == []


class TestSearchPatentsWithTools:
    def test_research_tool_results_returned(self):
        tool = ResearchTool(
            name="mock_search",
            description="Mock",
            fn=lambda q, ctx: [{"title": "Web Patent", "abstract": "Found via web"}],
        )
        results = search_patents("test problem", _mock_llm(), store=None, research_tools=[tool])
        assert len(results) == 1
        assert results[0]["title"] == "Web Patent"
        assert results[0]["source"] == "mock_search"

    def test_without_tools_unchanged(self):
        results = search_patents("test", _mock_llm(), store=None, research_tools=None)
        assert results == []

    def test_without_tools_and_store_returns_empty(self):
        results = search_patents("test", _mock_llm(), store=None)
        assert results == []

    def test_deduplicates_by_title(self):
        tool = ResearchTool(
            name="dup",
            description="Dup",
            fn=lambda q, ctx: [
                {"title": "Same Title", "abstract": "first"},
                {"title": "Same Title", "abstract": "second"},
            ],
        )
        results = search_patents("test", _mock_llm(), store=None, research_tools=[tool])
        assert len(results) == 1

    def test_tool_failure_continues(self):
        def failing_tool(q, ctx):
            raise RuntimeError("API down")

        tool = ResearchTool(name="broken", description="Broken", fn=failing_tool)
        results = search_patents("test", _mock_llm(), store=None, research_tools=[tool])
        assert results == []

    def test_results_normalized(self):
        tool = ResearchTool(
            name="minimal",
            description="Minimal",
            fn=lambda q, ctx: [{"title": "T", "abstract": "A"}],
        )
        results = search_patents("test", _mock_llm(), store=None, research_tools=[tool])
        r = results[0]
        assert r["id"] == ""
        assert r["matched_principles"] == []
        assert r["source"] == "minimal"

    def test_multiple_tools(self):
        tool1 = ResearchTool(
            name="tool1",
            description="Tool 1",
            fn=lambda q, ctx: [{"title": "Patent A", "abstract": "A"}],
        )
        tool2 = ResearchTool(
            name="tool2",
            description="Tool 2",
            fn=lambda q, ctx: [{"title": "Patent B", "abstract": "B"}],
        )
        results = search_patents("test", _mock_llm(), store=None, research_tools=[tool1, tool2])
        assert len(results) == 2
        assert {r["title"] for r in results} == {"Patent A", "Patent B"}

    def test_empty_title_skipped(self):
        tool = ResearchTool(
            name="empty",
            description="Empty",
            fn=lambda q, ctx: [{"title": "", "abstract": "no title"}],
        )
        results = search_patents("test", _mock_llm(), store=None, research_tools=[tool])
        assert results == []

    def test_partial_failure_keeps_good_results(self):
        def failing_tool(q, ctx):
            raise RuntimeError("API down")

        good_tool = ResearchTool(
            name="good",
            description="Good",
            fn=lambda q, ctx: [{"title": "Good Patent", "abstract": "Works"}],
        )
        bad_tool = ResearchTool(name="bad", description="Bad", fn=failing_tool)
        results = search_patents(
            "test", _mock_llm(), store=None, research_tools=[bad_tool, good_tool]
        )
        assert len(results) == 1
        assert results[0]["title"] == "Good Patent"

    def test_context_only_tool_skipped_in_search(self):
        """A tool with stages=["context"] should not run during search."""
        tool = ResearchTool(
            name="ctx_only",
            description="Context only",
            fn=lambda q, ctx: [{"title": "Should not appear", "abstract": "X"}],
            stages=["context"],
        )
        results = search_patents("test", _mock_llm(), store=None, research_tools=[tool])
        assert results == []

    def test_search_context_includes_params(self):
        """Search stage should receive principle_ids and param info in context."""
        captured = {}

        def capture_fn(q, ctx):
            captured.update(ctx)
            return [{"title": "Result", "abstract": "A"}]

        tool = ResearchTool(name="cap", description="Cap", fn=capture_fn)
        search_patents(
            "test",
            _mock_llm(),
            store=None,
            principle_ids=[1, 35],
            improving_param=5,
            worsening_param=10,
            research_tools=[tool],
        )
        assert captured["stage"] == "search"
        assert captured["principle_ids"] == [1, 35]
        assert captured["improving_param"] == 5
        assert captured["worsening_param"] == 10


class TestEnrichmentTools:
    def test_enrichment_tools_called_with_directions(self):
        tool = ResearchTool(
            name="costs",
            description="Cost estimates",
            fn=lambda q, ctx: [{"title": "Cost Analysis", "content": "$500/unit"}],
            stages=["enrichment"],
        )
        result = run_enrichment_tools("problem", [{"title": "Solution A"}], [tool])
        assert len(result) == 1
        assert result[0]["content"] == "$500/unit"

    def test_enrichment_receives_solution_directions(self):
        captured = {}

        def capture_fn(q, ctx):
            captured.update(ctx)
            return []

        tool = ResearchTool(name="t", description="T", fn=capture_fn, stages=["enrichment"])
        directions = [{"title": "Sol A"}, {"title": "Sol B"}]
        run_enrichment_tools("problem", directions, [tool])
        assert captured["stage"] == "enrichment"
        assert captured["solution_directions"] == directions

    def test_no_tools_returns_empty(self):
        assert run_enrichment_tools("problem", [], None) == []
        assert run_enrichment_tools("problem", [], []) == []

    def test_search_only_tool_skipped(self):
        tool = ResearchTool(
            name="search",
            description="Search",
            fn=lambda q, ctx: [{"title": "X", "content": "Y"}],
            stages=["search"],
        )
        result = run_enrichment_tools("problem", [], [tool])
        assert result == []
