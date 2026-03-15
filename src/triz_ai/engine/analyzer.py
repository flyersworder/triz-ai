"""TRIZ problem analysis pipeline."""

import logging

from pydantic import BaseModel

from triz_ai.knowledge.contradictions import lookup_with_observations
from triz_ai.knowledge.parameters import load_parameters
from triz_ai.knowledge.principles import load_principles
from triz_ai.llm.client import LLMClient
from triz_ai.patents.store import PatentStore

logger = logging.getLogger(__name__)


class AnalysisResult(BaseModel):
    """Result of TRIZ analysis of a problem."""

    problem: str
    improving_param: dict  # {"id": int, "name": str}
    worsening_param: dict  # {"id": int, "name": str}
    reasoning: str
    contradiction_confidence: float = 1.0
    recommended_principles: list[dict]  # [{"id": int, "name": str, "description": str}]
    patent_examples: list[dict]
    solution_directions: list[dict] = []


def analyze(
    problem_text: str,
    llm_client: LLMClient | None = None,
    store: PatentStore | None = None,
) -> AnalysisResult:
    """Analyze a technical problem using TRIZ methodology.

    Pipeline:
    1. LLM extracts the technical contradiction
    2. Maps to engineering parameters
    3. Looks up contradiction matrix for recommended principles
    4. Searches patent store for examples
    5. Returns structured result
    """
    if llm_client is None:
        llm_client = LLMClient()

    # Step 1: Extract contradiction
    contradiction = llm_client.extract_contradiction(problem_text)

    # Step 2: Map to parameters
    parameters = {p.id: p for p in load_parameters()}
    improving = parameters.get(contradiction.improving_param)
    worsening = parameters.get(contradiction.worsening_param)

    if not improving or not worsening:
        raise ValueError(
            f"Invalid parameters: improving={contradiction.improving_param}, "
            f"worsening={contradiction.worsening_param}"
        )

    # Step 3: Lookup matrix (merges static + patent observations when store available)
    principle_ids = lookup_with_observations(
        contradiction.improving_param, contradiction.worsening_param, store=store
    )

    # Map to principle details
    all_principles = {p.id: p for p in load_principles()}
    recommended_principles = []
    for pid in principle_ids:
        p = all_principles.get(pid)
        if p:
            recommended_principles.append(
                {"id": p.id, "name": p.name, "description": p.description}
            )

    # Step 4: Hybrid patent search (if store available)
    patent_examples = []
    recommended_principle_ids = [p["id"] for p in recommended_principles]
    if store is not None:
        try:
            query_embedding = llm_client.get_embedding(problem_text)
            results = store.search_patents_hybrid(
                query_embedding,
                principle_ids=recommended_principle_ids,
                improving_param=contradiction.improving_param,
                worsening_param=contradiction.worsening_param,
                limit=5,
            )
            # Enrich with assignee, filing_date, matched_principles
            all_principles_map = {p.id: p for p in load_principles()}
            for patent, _score in results:
                matched_principles = []
                classification = store.get_classification(patent.id)
                if classification:
                    overlap = set(recommended_principle_ids) & set(classification.principle_ids)
                    matched_principles = [
                        all_principles_map[pid].name
                        for pid in overlap
                        if pid in all_principles_map
                    ]
                patent_examples.append(
                    {
                        "id": patent.id,
                        "title": patent.title,
                        "abstract": patent.abstract or "",
                        "assignee": patent.assignee,
                        "filing_date": patent.filing_date,
                        "matched_principles": matched_principles,
                    }
                )
        except Exception:
            logger.warning("Patent search failed, continuing without examples")

    # Step 5: Generate solution directions
    solution_directions = []
    if recommended_principles:
        try:
            directions = llm_client.generate_solution_directions(
                problem_text,
                improving_param=improving.name,
                worsening_param=worsening.name,
                principles=recommended_principles,
                patent_examples=patent_examples,
            )
            solution_directions = [d.model_dump() for d in directions.directions]
        except Exception:
            logger.warning("Solution direction generation failed, continuing without")

    return AnalysisResult(
        problem=problem_text,
        improving_param={"id": improving.id, "name": improving.name},
        worsening_param={"id": worsening.id, "name": worsening.name},
        reasoning=contradiction.reasoning,
        contradiction_confidence=contradiction.confidence,
        recommended_principles=recommended_principles,
        patent_examples=patent_examples,
        solution_directions=solution_directions,
    )
