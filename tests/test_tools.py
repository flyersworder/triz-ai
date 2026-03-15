"""Tests for pluggable research tools interface."""

from unittest.mock import MagicMock

from triz_ai.engine.analyzer import search_patents
from triz_ai.tools import ResearchTool


def _mock_llm():
    client = MagicMock()
    client.get_embedding.return_value = [0.1] * 768
    return client


class TestResearchToolDataclass:
    def test_fields(self):
        tool = ResearchTool(name="test", description="A test tool", fn=lambda q: [])
        assert tool.name == "test"
        assert tool.description == "A test tool"
        assert callable(tool.fn)


class TestSearchPatentsWithTools:
    def test_research_tool_results_returned(self):
        tool = ResearchTool(
            name="mock_search",
            description="Mock",
            fn=lambda q: [{"title": "Web Patent", "abstract": "Found via web"}],
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
            fn=lambda q: [
                {"title": "Same Title", "abstract": "first"},
                {"title": "Same Title", "abstract": "second"},
            ],
        )
        results = search_patents("test", _mock_llm(), store=None, research_tools=[tool])
        assert len(results) == 1

    def test_tool_failure_continues(self):
        def failing_tool(q):
            raise RuntimeError("API down")

        tool = ResearchTool(name="broken", description="Broken", fn=failing_tool)
        results = search_patents("test", _mock_llm(), store=None, research_tools=[tool])
        assert results == []

    def test_results_normalized(self):
        tool = ResearchTool(
            name="minimal",
            description="Minimal",
            fn=lambda q: [{"title": "T", "abstract": "A"}],
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
            fn=lambda q: [{"title": "Patent A", "abstract": "A"}],
        )
        tool2 = ResearchTool(
            name="tool2",
            description="Tool 2",
            fn=lambda q: [{"title": "Patent B", "abstract": "B"}],
        )
        results = search_patents("test", _mock_llm(), store=None, research_tools=[tool1, tool2])
        assert len(results) == 2
        assert {r["title"] for r in results} == {"Patent A", "Patent B"}

    def test_empty_title_skipped(self):
        tool = ResearchTool(
            name="empty",
            description="Empty",
            fn=lambda q: [{"title": "", "abstract": "no title"}],
        )
        results = search_patents("test", _mock_llm(), store=None, research_tools=[tool])
        assert results == []

    def test_partial_failure_keeps_good_results(self):
        def failing_tool(q):
            raise RuntimeError("API down")

        good_tool = ResearchTool(
            name="good",
            description="Good",
            fn=lambda q: [{"title": "Good Patent", "abstract": "Works"}],
        )
        bad_tool = ResearchTool(name="bad", description="Bad", fn=failing_tool)
        results = search_patents(
            "test", _mock_llm(), store=None, research_tools=[bad_tool, good_tool]
        )
        assert len(results) == 1
        assert results[0]["title"] == "Good Patent"
