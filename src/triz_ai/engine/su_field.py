"""Su-Field analysis pipeline."""

import logging

from triz_ai.engine.analyzer import AnalysisResult, search_patents
from triz_ai.llm.client import LLMClient
from triz_ai.patents.store import PatentStore

logger = logging.getLogger(__name__)


def analyze_su_field(
    problem_text: str,
    ideal_final_result: str | None,
    llm_client: LLMClient,
    store: PatentStore | None = None,
    research_tools: list | None = None,
) -> AnalysisResult:
    """Su-Field analysis pipeline.

    1. Identify substances and field in the system
    2. Classify the problem type (incomplete/harmful/inefficient)
    3. Recommend standard solutions
    4. Search patents for examples
    5. Generate solution directions
    """
    result = llm_client.analyze_su_field(problem_text)

    patent_examples = search_patents(
        problem_text, llm_client, store, research_tools=research_tools
    )

    # Generate solution directions from standard solutions
    solution_directions = []
    try:
        principles = [
            {"name": ss["name"], "description": ss.get("applicability", "")}
            for ss in result.standard_solutions
        ]
        if principles:
            directions = llm_client.generate_solution_directions(
                problem_text,
                improving_param=f"Su-Field interaction ({result.field})",
                worsening_param=f"Problem: {result.problem_type}",
                principles=principles,
                patent_examples=patent_examples,
            )
            solution_directions = [d.model_dump() for d in directions.directions]
    except Exception:
        logger.warning("Solution direction generation failed, continuing without")

    return AnalysisResult(
        problem=problem_text,
        method="su_field",
        ideal_final_result=ideal_final_result,
        reasoning=f"Su-Field model: {', '.join(result.substances)} interacting via "
        f"{result.field} — problem type: {result.problem_type}",
        patent_examples=patent_examples,
        solution_directions=solution_directions,
        details={
            "substances": result.substances,
            "field": result.field,
            "problem_type": result.problem_type,
            "standard_solutions": result.standard_solutions,
        },
    )
