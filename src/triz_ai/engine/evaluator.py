"""Idea evaluation against prior art."""

import logging

from pydantic import BaseModel

from triz_ai.llm.client import LLMClient
from triz_ai.patents.store import PatentStore

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    """Result of evaluating an idea against prior art."""

    idea: str
    domain: str
    novelty_score: float  # 0-1, higher = more novel
    similar_patents: list[dict]  # [{"id": str, "title": str, "similarity": float}]
    assessment: str


def evaluate(
    idea: str,
    domain: str,
    llm_client: LLMClient | None = None,
    store: PatentStore | None = None,
) -> EvaluationResult:
    """Evaluate an idea against existing patents in the store.

    Searches for similar existing patents and scores novelty based on
    how different the idea is from existing work.
    """
    if llm_client is None:
        llm_client = LLMClient()
    if store is None:
        store = PatentStore()
        store.init_db()

    # Search for similar patents
    similar_patents = []
    try:
        query_embedding = llm_client.get_embedding(idea)
        results = store.search_patents(query_embedding, limit=5)
        for patent, distance in results:
            # Convert distance to similarity (lower distance = higher similarity)
            similarity = max(0.0, 1.0 - distance)
            similar_patents.append(
                {"id": patent.id, "title": patent.title, "similarity": round(similarity, 3)}
            )
    except Exception:
        logger.warning("Patent search failed during evaluation")

    # Calculate novelty score based on most similar patent
    if similar_patents:
        max_similarity = max(p["similarity"] for p in similar_patents)
        novelty_score = 1.0 - max_similarity
    else:
        novelty_score = 1.0  # No prior art found = fully novel

    # Generate assessment
    if novelty_score > 0.8:
        assessment = "Highly novel — no close prior art found"
    elif novelty_score > 0.5:
        assessment = "Moderately novel — some related prior art exists"
    elif novelty_score > 0.3:
        assessment = "Limited novelty — similar work exists in the patent store"
    else:
        assessment = "Low novelty — very similar patents already exist"

    return EvaluationResult(
        idea=idea,
        domain=domain,
        novelty_score=round(novelty_score, 3),
        similar_patents=similar_patents,
        assessment=assessment,
    )
