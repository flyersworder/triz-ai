"""Trimming analysis pipeline."""

import logging

from triz_ai.engine.analyzer import AnalysisResult, run_enrichment_tools, search_patents
from triz_ai.llm.client import LLMClient
from triz_ai.patents.store import PatentStore

logger = logging.getLogger(__name__)


def analyze_trimming(
    problem_text: str,
    ideal_final_result: str | None,
    llm_client: LLMClient,
    store: PatentStore | None = None,
    research_tools: list | None = None,
) -> AnalysisResult:
    """Trimming analysis pipeline.

    1. Decompose system into components with cost assessment
    2. Identify trimming candidates
    3. Show how functions are redistributed
    4. Search patents for examples
    5. Generate solution directions
    6. Run enrichment-stage research tools
    """
    result = llm_client.analyze_trimming(problem_text)

    patent_examples = search_patents(
        problem_text, llm_client, store, research_tools=research_tools
    )

    # Generate solution directions
    solution_directions = []
    try:
        if result.trimming_candidates:
            principles = [
                {
                    "name": f"Trim {tc['component']}",
                    "description": f"{tc['reason']} (Rule {tc['rule']})",
                }
                for tc in result.trimming_candidates
            ]
            directions = llm_client.generate_solution_directions(
                problem_text,
                improving_param="system simplicity / cost",
                worsening_param="component count / complexity",
                principles=principles,
                patent_examples=patent_examples,
            )
            solution_directions = [d.model_dump() for d in directions.directions]
    except Exception:
        logger.warning("Solution direction generation failed, continuing without")

    enrichment = run_enrichment_tools(problem_text, solution_directions, research_tools)

    return AnalysisResult(
        problem=problem_text,
        method="trimming",
        ideal_final_result=ideal_final_result,
        reasoning=f"Trimming analysis identified {len(result.trimming_candidates)} "
        f"candidate(s) for removal out of {len(result.components)} components",
        patent_examples=patent_examples,
        solution_directions=solution_directions,
        enrichment=enrichment,
        details={
            "components": result.components,
            "trimming_candidates": result.trimming_candidates,
            "redistributed_functions": result.redistributed_functions,
        },
    )
