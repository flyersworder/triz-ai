"""Usage-driven self-evolution — learn from web search results during analysis."""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from triz_ai.engine.analyzer import AnalysisResult
    from triz_ai.patents.repository import PatentRepository

logger = logging.getLogger(__name__)


class SearchObservation(BaseModel):
    """A web search result captured during analysis, with its analysis context."""

    id: str
    title: str
    snippet: str | None = None
    url: str | None = None
    source_tool: str | None = None
    problem_text: str | None = None
    analysis_method: str | None = None
    improving_param: int | None = None
    worsening_param: int | None = None
    principle_ids: list[int] = []
    analysis_confidence: float = 0.0
    consolidated: bool = False
    observed_at: str | None = None
    consolidated_at: str | None = None


class ConsolidationResult(BaseModel):
    """Summary of a consolidation run."""

    observations_processed: int = 0
    matrix_observations_added: int = 0
    candidate_principles_proposed: int = 0
    candidate_parameters_proposed: int = 0
    observations_pruned: int = 0


def _make_observation_id(title: str, snippet: str | None) -> str:
    """Generate a deterministic ID for deduplication."""
    content = f"{title}|{snippet or ''}"
    hash_hex = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"ws:{hash_hex}"


def collect_search_observations(
    result: AnalysisResult,
    store: PatentRepository,
) -> int:
    """Store web search results from analysis as search observations.

    Filters patent_examples to those with a 'source' field (web results),
    builds a SearchObservation from each, and stores in the DB.

    Returns number of observations stored.
    """
    count = 0
    for example in result.patent_examples:
        source = example.get("source")
        if not source:
            continue

        title = example.get("title", "")
        snippet = example.get("abstract", "")
        if not title:
            continue

        obs = SearchObservation(
            id=_make_observation_id(title, snippet),
            title=title,
            snippet=snippet,
            url=example.get("url"),
            source_tool=source,
            problem_text=result.problem,
            analysis_method=result.method,
            improving_param=(result.improving_param["id"] if result.improving_param else None),
            worsening_param=(result.worsening_param["id"] if result.worsening_param else None),
            principle_ids=[p["id"] for p in result.recommended_principles],
            analysis_confidence=result.contradiction_confidence,
            observed_at=datetime.now(UTC).isoformat(),
        )
        store.insert_search_observation(obs)
        count += 1

    if count > 0:
        store.increment_analysis_count()
        logger.debug("Collected %d search observations", count)

    return count
