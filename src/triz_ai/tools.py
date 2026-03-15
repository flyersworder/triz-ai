"""Pluggable research tools for supplementing patent search."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

VALID_STAGES = {"context", "search", "enrichment"}


@dataclass
class ResearchTool:
    """A research tool that supplements built-in analysis at specific stages.

    Developers pass these to route() or orchestrate_deep() to supplement
    the built-in patent DB search and enrich analysis at multiple stages.

    Args:
        name: Identifier (e.g. "google_patents", "arxiv").
        description: Shown to LLM in deep mode so it can decide whether
            to use this tool. Be specific about what the tool searches
            and when it's most useful.
        fn: Callable(query: str, context: dict) -> list[dict].
            context includes {"stage": str} plus stage-specific data.
        stages: Which pipeline stages this tool runs at.
            "context" — before LLM extraction; return [{"content": "..."}]
            "search" — during patent search; return [{"title": "...", "abstract": "..."}]
            "enrichment" — after solution generation; return [{"title": "...", "content": "..."}]
            Defaults to ["search"] for backward compatibility.
    """

    name: str
    description: str
    fn: Callable[[str, dict], list[dict]]
    stages: list[str] = field(default_factory=lambda: ["search"])

    def __post_init__(self):
        invalid = set(self.stages) - VALID_STAGES
        if invalid:
            raise ValueError(
                f"Invalid stages {invalid} for tool '{self.name}'. "
                f"Valid stages: {', '.join(sorted(VALID_STAGES))}"
            )


def run_stage_tools(
    tools: list[ResearchTool] | None,
    stage: str,
    query: str,
    extra_context: dict | None = None,
) -> list[dict]:
    """Run all tools registered for the given stage, collecting results.

    Tools that fail are logged and skipped — they never block analysis.
    """
    if not tools:
        return []
    context = {"stage": stage}
    if extra_context:
        context.update(extra_context)
    results = []
    for tool in tools:
        if stage not in tool.stages:
            continue
        try:
            results.extend(tool.fn(query, context))
        except Exception:
            logger.warning("Research tool '%s' failed at stage '%s', skipping", tool.name, stage)
    return results
