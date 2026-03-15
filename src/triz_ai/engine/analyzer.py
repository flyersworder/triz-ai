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
    """Result of TRIZ analysis of a problem.

    The result is tool-agnostic at the top level, with tool-specific data in `details`.
    For backward compatibility, contradiction-specific fields are kept but optional.
    """

    problem: str
    method: str = "technical_contradiction"
    method_confidence: float = 1.0
    secondary_method: str | None = None
    ideal_final_result: str | None = None

    # Contradiction-specific (kept for backward compat, populated from details)
    improving_param: dict | None = None  # {"id": int, "name": str}
    worsening_param: dict | None = None  # {"id": int, "name": str}
    reasoning: str = ""
    contradiction_confidence: float = 1.0
    recommended_principles: list[dict] = []  # [{"id": int, "name": str, "description": str}]

    # Common across all methods
    patent_examples: list[dict] = []
    solution_directions: list[dict] = []
    details: dict = {}  # Tool-specific data


def analyze_contradiction(
    problem_text: str,
    ideal_final_result: str | None,
    llm_client: LLMClient,
    store: PatentStore | None = None,
) -> AnalysisResult:
    """Technical contradiction analysis pipeline.

    Pipeline:
    1. LLM extracts the technical contradiction
    2. Maps to engineering parameters
    3. Looks up contradiction matrix for recommended principles
    4. Searches patent store for examples
    5. Generates solution directions
    """
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
    patent_examples = search_patents(
        problem_text,
        llm_client,
        store,
        principle_ids=[p["id"] for p in recommended_principles],
        improving_param=contradiction.improving_param,
        worsening_param=contradiction.worsening_param,
    )

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

    improving_dict = {"id": improving.id, "name": improving.name}
    worsening_dict = {"id": worsening.id, "name": worsening.name}

    return AnalysisResult(
        problem=problem_text,
        method="technical_contradiction",
        ideal_final_result=ideal_final_result,
        improving_param=improving_dict,
        worsening_param=worsening_dict,
        reasoning=contradiction.reasoning,
        contradiction_confidence=contradiction.confidence,
        recommended_principles=recommended_principles,
        patent_examples=patent_examples,
        solution_directions=solution_directions,
        details={
            "improving_param": improving_dict,
            "worsening_param": worsening_dict,
            "reasoning": contradiction.reasoning,
            "contradiction_confidence": contradiction.confidence,
            "recommended_principles": recommended_principles,
        },
    )


def search_patents(
    problem_text: str,
    llm_client: LLMClient,
    store: PatentStore | None,
    principle_ids: list[int] | None = None,
    improving_param: int | None = None,
    worsening_param: int | None = None,
) -> list[dict]:
    """Search patent store for relevant examples."""
    if store is None:
        return []

    try:
        query_embedding = llm_client.get_embedding(problem_text)
        if principle_ids and improving_param and worsening_param:
            results = store.search_patents_hybrid(
                query_embedding,
                principle_ids=principle_ids,
                improving_param=improving_param,
                worsening_param=worsening_param,
                limit=5,
            )
        else:
            results = store.search_patents(query_embedding, limit=5)

        all_principles_map = {p.id: p for p in load_principles()}
        patent_examples = []
        for patent, _score in results:
            matched_principles = []
            if principle_ids:
                classification = store.get_classification(patent.id)
                if classification:
                    overlap = set(principle_ids) & set(classification.principle_ids)
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
        return patent_examples
    except Exception:
        logger.warning("Patent search failed, continuing without examples")
        return []


def analyze(
    problem_text: str,
    llm_client: LLMClient | None = None,
    store: PatentStore | None = None,
) -> AnalysisResult:
    """Analyze a technical problem using TRIZ methodology.

    Legacy entry point — delegates to analyze_contradiction for backward compatibility.
    """
    if llm_client is None:
        llm_client = LLMClient()

    return analyze_contradiction(problem_text, None, llm_client, store)
