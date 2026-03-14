"""White space / idea generation using underused TRIZ principles."""

import logging
from collections import Counter

from pydantic import BaseModel

from triz_ai.knowledge.principles import load_principles
from triz_ai.llm.client import LLMClient
from triz_ai.patents.store import PatentStore

logger = logging.getLogger(__name__)


class DiscoveryReport(BaseModel):
    """Report of principle usage and generated ideas for a domain."""

    domain: str
    total_patents: int
    principle_usage: list[dict]  # [{"id": int, "name": str, "count": int}]
    underused_principles: list[dict]  # [{"id": int, "name": str, "description": str}]
    ideas: list[dict]  # [{"idea": str, "principle_id": int, "reasoning": str}]


def discover(
    domain: str,
    llm_client: LLMClient | None = None,
    store: PatentStore | None = None,
) -> DiscoveryReport:
    """Discover underused TRIZ principles and generate ideas for a domain.

    Pipeline:
    1. Query classifications by domain
    2. Aggregate principle usage statistics
    3. Identify underused principles
    4. Generate novel ideas using underused principles
    """
    if llm_client is None:
        llm_client = LLMClient()
    if store is None:
        store = PatentStore()
        store.init_db()

    # Step 1: Get classifications for domain
    classified = store.get_classifications_by_domain(domain)
    total_patents = len(classified)

    # Step 2: Aggregate principle usage
    all_principles = {p.id: p for p in load_principles()}
    usage_counter: Counter[int] = Counter()
    for _patent, classification in classified:
        for pid in classification.principle_ids:
            usage_counter[pid] += 1

    # Build usage table (all 40 principles)
    principle_usage = []
    for pid in sorted(all_principles.keys()):
        p = all_principles[pid]
        principle_usage.append({"id": pid, "name": p.name, "count": usage_counter.get(pid, 0)})

    # Step 3: Identify underused principles (used less than average or not at all)
    if total_patents > 0:
        avg_usage = sum(usage_counter.values()) / max(len(usage_counter), 1)
        underused = [
            {
                "id": pid,
                "name": all_principles[pid].name,
                "description": all_principles[pid].description,
            }
            for pid in sorted(all_principles.keys())
            if usage_counter.get(pid, 0) < max(avg_usage * 0.5, 1)
        ]
    else:
        # No data — all principles are "underused"
        underused = [
            {"id": pid, "name": p.name, "description": p.description}
            for pid, p in sorted(all_principles.items())
        ]

    # Step 4: Generate ideas (limit underused to top 10 for token budget)
    existing_titles = [patent.title for patent, _ in classified[:10]]
    ideas = []
    if underused:
        try:
            idea_batch = llm_client.generate_ideas(
                domain=domain,
                underused_principles=underused[:10],
                existing_patents=existing_titles,
            )
            ideas = [idea.model_dump() for idea in idea_batch.ideas]
        except Exception:
            logger.exception("Idea generation failed")

    return DiscoveryReport(
        domain=domain,
        total_patents=total_patents,
        principle_usage=principle_usage,
        underused_principles=underused,
        ideas=ideas,
    )
