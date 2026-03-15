"""Pluggable research tools for supplementing patent search."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ResearchTool:
    """A research tool that can find prior art or technical references.

    Developers pass these to route() or orchestrate_deep() to supplement
    the built-in patent DB search.

    Args:
        name: Identifier (e.g. "google_patents", "arxiv").
        description: Shown to LLM in deep mode so it can decide whether
            to use this tool. Be specific about what the tool searches
            and when it's most useful.
        fn: Callable that takes a search query string and returns results.
            Each result dict must have at least "title" and "abstract".
            Optional fields: "id", "assignee", "filing_date", "url",
            "matched_principles".
    """

    name: str
    description: str
    fn: Callable[[str], list[dict]]
