"""Usage-driven self-evolution — learn from web search results during analysis."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    pass

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
