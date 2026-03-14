"""Patent TRIZ classification."""

import logging

from pydantic import BaseModel

from triz_ai.llm.client import LLMClient
from triz_ai.patents.store import Classification, PatentStore

logger = logging.getLogger(__name__)


class ClassificationResult(BaseModel):
    """Result of classifying a patent through TRIZ lens."""

    patent_id: str | None
    principle_ids: list[int]
    contradiction: dict
    confidence: float
    reasoning: str


def classify(
    patent_text: str,
    patent_id: str | None = None,
    llm_client: LLMClient | None = None,
    store: PatentStore | None = None,
) -> ClassificationResult:
    """Classify a patent by TRIZ principles.

    Args:
        patent_text: The patent text to classify.
        patent_id: Optional patent ID for storing classification.
        llm_client: LLM client (created with defaults if None).
        store: Patent store to save classification to (optional).

    Returns:
        ClassificationResult with principle IDs, contradiction, and confidence.
    """
    if llm_client is None:
        llm_client = LLMClient()

    classification = llm_client.classify_patent(patent_text)

    result = ClassificationResult(
        patent_id=patent_id,
        principle_ids=classification.principle_ids,
        contradiction=classification.contradiction,
        confidence=classification.confidence,
        reasoning=classification.reasoning,
    )

    # Store if we have both a store and a patent_id
    if store is not None and patent_id is not None:
        store.insert_classification(
            Classification(
                patent_id=patent_id,
                principle_ids=classification.principle_ids,
                contradiction=classification.contradiction,
                confidence=classification.confidence,
            )
        )
        logger.info("Stored classification for patent %s", patent_id)

    return result
