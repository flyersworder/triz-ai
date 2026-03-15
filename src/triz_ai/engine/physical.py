"""Physical contradiction analysis pipeline."""

import logging

from triz_ai.engine.analyzer import AnalysisResult, run_enrichment_tools, search_patents
from triz_ai.llm.client import LLMClient
from triz_ai.patents.store import PatentStore

logger = logging.getLogger(__name__)


def analyze_physical(
    problem_text: str,
    ideal_final_result: str | None,
    llm_client: LLMClient,
    store: PatentStore | None = None,
    research_tools: list | None = None,
) -> AnalysisResult:
    """Physical contradiction analysis pipeline.

    1. Extract the physical contradiction (property + opposing requirements)
    2. Recommend separation principles
    3. Search patents for examples
    4. Generate solution directions
    5. Run enrichment-stage research tools
    """
    result = llm_client.extract_physical_contradiction(problem_text)

    # Patent search (vector-only, no contradiction-specific bonuses)
    patent_examples = search_patents(
        problem_text, llm_client, store, research_tools=research_tools
    )

    # Generate solution directions
    solution_directions = []
    try:
        directions = llm_client.generate_solution_directions(
            problem_text,
            improving_param=result.requirement_a,
            worsening_param=result.requirement_b,
            principles=[
                {"name": sp["name"], "description": sp.get("technique", "")}
                for sp in result.separation_principles
            ],
            patent_examples=patent_examples,
        )
        solution_directions = [d.model_dump() for d in directions.directions]
    except Exception:
        logger.warning("Solution direction generation failed, continuing without")

    enrichment = run_enrichment_tools(problem_text, solution_directions, research_tools)

    return AnalysisResult(
        problem=problem_text,
        method="physical_contradiction",
        ideal_final_result=ideal_final_result,
        reasoning=f"Physical contradiction: {result.property} must be "
        f"'{result.requirement_a}' AND '{result.requirement_b}'",
        patent_examples=patent_examples,
        solution_directions=solution_directions,
        enrichment=enrichment,
        details={
            "property": result.property,
            "requirement_a": result.requirement_a,
            "requirement_b": result.requirement_b,
            "separation_type": result.separation_type,
            "separation_principles": result.separation_principles,
        },
    )
