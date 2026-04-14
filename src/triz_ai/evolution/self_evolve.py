"""Usage-driven self-evolution — learn from web search results during analysis."""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from triz_ai.engine.analyzer import AnalysisResult
    from triz_ai.llm.client import LLMClient
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
        logger.debug("Collected %d search observations", count)

    return count


def maybe_auto_consolidate(
    llm_client: LLMClient,
    store: PatentRepository,
    consolidation_interval: int | None = None,
) -> ConsolidationResult | None:
    """Auto-consolidate if analysis count exceeds threshold.

    Args:
        consolidation_interval: Override threshold (default: from config).

    Returns ConsolidationResult if consolidation ran, None otherwise.
    """
    if consolidation_interval is None:
        from triz_ai.config import load_config

        consolidation_interval = load_config().evolution.consolidation_interval

    count = store.get_analyses_since_consolidation()
    if count < consolidation_interval:
        return None

    result = consolidate(llm_client, store)
    store.reset_analysis_count()
    return result


def consolidate(
    llm_client: LLMClient,
    store: PatentRepository,
    retention_days: int | None = None,
    min_observations: int = 3,
    source_confidence_weight: float | None = None,
) -> ConsolidationResult:
    """Consolidate search observations into matrix observations and candidates.

    Steps:
    1. Load unconsolidated observations
    2. Group by (improving_param, worsening_param)
    3. LLM validates principle assignments per group
    4. Record matrix observations (with source confidence discount)
    5. Cluster low-confidence observations for candidate discovery
    6. Mark consolidated and prune
    """
    from triz_ai.config import load_config
    from triz_ai.knowledge.parameters import get_parameter

    config = load_config()
    if retention_days is None:
        retention_days = config.evolution.retention_days
    if source_confidence_weight is None:
        source_confidence_weight = config.evolution.source_confidence_weight

    observations = store.get_unconsolidated_observations()
    if not observations:
        return ConsolidationResult()

    # Group by contradiction pair
    groups: dict[tuple[int | None, int | None], list[SearchObservation]] = {}
    for obs in observations:
        key = (obs.improving_param, obs.worsening_param)
        if key not in groups:
            groups[key] = []
        groups[key].append(obs)

    matrix_obs_added = 0
    all_low_confidence: list[SearchObservation] = []

    for (improving, worsening), group_obs in groups.items():
        # Skip non-contradiction groups for matrix recording
        if improving is None or worsening is None:
            all_low_confidence.extend(group_obs)
            continue

        # Collect principle IDs from all observations in this group
        principle_set: set[int] = set()
        for obs in group_obs:
            principle_set.update(obs.principle_ids)

        if not principle_set:
            all_low_confidence.extend(group_obs)
            continue

        # Get parameter names for the LLM prompt
        imp_param = get_parameter(improving)
        wor_param = get_parameter(worsening)
        imp_name = imp_param.name if imp_param else f"Parameter {improving}"
        wor_name = wor_param.name if wor_param else f"Parameter {worsening}"

        # LLM validates principle assignments
        try:
            validation = llm_client.validate_observations(
                observations=[
                    {"id": o.id, "title": o.title, "snippet": o.snippet} for o in group_obs
                ],
                improving_param=improving,
                improving_name=imp_name,
                worsening_param=worsening,
                worsening_name=wor_name,
                principle_ids=sorted(principle_set),
            )
        except Exception:
            logger.warning(
                "Observation validation failed for (%d, %d), skipping",
                improving,
                worsening,
            )
            continue

        # Aggregate validated confidence per principle
        principle_scores: dict[int, list[float]] = {}
        low_conf_obs_ids: set[str] = set()
        valid_obs_ids = {o.id for o in group_obs}

        for v in validation.validations:
            if v.observation_id not in valid_obs_ids:
                continue  # LLM hallucinated an ID — skip
            has_high_conf = False
            for vp in v.validated_principles:
                if vp.confidence >= config.evolution.review_threshold:
                    has_high_conf = True
                    if vp.principle_id not in principle_scores:
                        principle_scores[vp.principle_id] = []
                    principle_scores[vp.principle_id].append(vp.confidence)

            if not has_high_conf:
                low_conf_obs_ids.add(v.observation_id)

        # Record matrix observations for principles with enough evidence
        for principle_id, scores in principle_scores.items():
            if len(scores) >= min_observations:
                avg_conf = sum(scores) / len(scores)
                weighted_conf = avg_conf * source_confidence_weight
                for obs in group_obs:
                    if obs.id not in low_conf_obs_ids:
                        store.insert_matrix_observation(
                            improving=improving,
                            worsening=worsening,
                            principle_id=principle_id,
                            patent_id=obs.id,
                            confidence=weighted_conf,
                        )
                        matrix_obs_added += 1

        # Collect low-confidence observations for candidate discovery
        for obs in group_obs:
            if obs.id in low_conf_obs_ids:
                all_low_confidence.append(obs)

    # Candidate principle discovery from low-confidence observations
    candidates_proposed = 0
    if len(all_low_confidence) >= min_observations:
        try:
            snippets = [f"{o.title}\n{o.snippet or ''}" for o in all_low_confidence]
            clusters = llm_client.cluster_patents(snippets)
            for cluster_indices in clusters:
                if len(cluster_indices) < min_observations:
                    continue
                cluster_texts = [snippets[i] for i in cluster_indices if i < len(snippets)]
                if len(cluster_texts) < min_observations:
                    continue
                try:
                    proposal = llm_client.propose_candidate_principle(cluster_texts)
                    from triz_ai.patents.store import CandidatePrinciple

                    next_id = store.get_next_candidate_id()
                    candidate = CandidatePrinciple(
                        id=f"C{next_id}",
                        name=proposal.name,
                        description=proposal.description,
                        evidence_patent_ids=[
                            all_low_confidence[i].id
                            for i in cluster_indices
                            if i < len(all_low_confidence)
                        ],
                        confidence=proposal.confidence,
                    )
                    store.insert_candidate_principle(candidate)
                    candidates_proposed += 1
                    logger.info(
                        "Proposed candidate principle from web observations: %s — %s",
                        candidate.id,
                        candidate.name,
                    )
                except Exception:
                    logger.warning("Failed to propose candidate for cluster, skipping")
        except Exception:
            logger.warning("Clustering low-confidence observations failed, skipping")

    # Mark all as consolidated and prune
    store.mark_observations_consolidated([o.id for o in observations])
    pruned = store.prune_observations(retention_days=retention_days)

    result = ConsolidationResult(
        observations_processed=len(observations),
        matrix_observations_added=matrix_obs_added,
        candidate_principles_proposed=candidates_proposed,
        observations_pruned=pruned,
    )
    logger.info(
        "Consolidation complete: %d processed, %d matrix obs added, "
        "%d candidates proposed, %d pruned",
        result.observations_processed,
        result.matrix_observations_added,
        result.candidate_principles_proposed,
        result.observations_pruned,
    )
    return result
