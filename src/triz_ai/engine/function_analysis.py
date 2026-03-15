"""Function analysis pipeline."""

import logging

from triz_ai.engine.analyzer import AnalysisResult, search_patents
from triz_ai.llm.client import LLMClient
from triz_ai.patents.store import PatentStore

logger = logging.getLogger(__name__)


def analyze_functions(
    problem_text: str,
    ideal_final_result: str | None,
    llm_client: LLMClient,
    store: PatentStore | None = None,
    research_tools: list | None = None,
) -> AnalysisResult:
    """Function analysis pipeline.

    1. Decompose system into components and functions
    2. Identify problematic functions (harmful/insufficient/excessive)
    3. Generate recommendations
    4. Search patents for examples
    5. Generate solution directions
    """
    result = llm_client.analyze_functions(problem_text)

    patent_examples = search_patents(
        problem_text, llm_client, store, research_tools=research_tools
    )

    # Build solution directions from recommendations
    solution_directions = []
    try:
        if result.recommendations:
            principles = [
                {"name": f"Recommendation {i + 1}", "description": rec}
                for i, rec in enumerate(result.recommendations)
            ]
            directions = llm_client.generate_solution_directions(
                problem_text,
                improving_param="system functionality",
                worsening_param="harmful/insufficient functions",
                principles=principles,
                patent_examples=patent_examples,
            )
            solution_directions = [d.model_dump() for d in directions.directions]
    except Exception:
        logger.warning("Solution direction generation failed, continuing without")

    problem_summary = (
        "; ".join(
            f"{pf['subject']} {pf['action']} {pf['object']}: {pf['problem']}"
            for pf in result.problem_functions
        )
        if result.problem_functions
        else "No problematic functions identified"
    )

    return AnalysisResult(
        problem=problem_text,
        method="function_analysis",
        ideal_final_result=ideal_final_result,
        reasoning=f"Function analysis identified {len(result.problem_functions)} "
        f"problematic function(s): {problem_summary}",
        patent_examples=patent_examples,
        solution_directions=solution_directions,
        details={
            "components": result.components,
            "functions": result.functions,
            "problem_functions": result.problem_functions,
            "recommendations": result.recommendations,
        },
    )
