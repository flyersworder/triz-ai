"""Technology evolution trends + system operator pipeline."""

import logging

from triz_ai.engine.analyzer import AnalysisResult, search_patents
from triz_ai.llm.client import LLMClient
from triz_ai.patents.store import PatentStore

logger = logging.getLogger(__name__)


def analyze_trends(
    problem_text: str,
    ideal_final_result: str | None,
    llm_client: LLMClient,
    store: PatentStore | None = None,
    research_tools: list | None = None,
) -> AnalysisResult:
    """Trends + system operator analysis pipeline.

    1. Identify current position on evolution trend
    2. Predict next evolutionary stages
    3. Apply system operator (subsystem/system/supersystem × past/present/future)
    4. Search patents for examples
    5. Generate predictions
    """
    result = llm_client.analyze_trends(problem_text)

    patent_examples = search_patents(
        problem_text, llm_client, store, research_tools=research_tools
    )

    # Generate solution directions from predictions
    solution_directions = []
    try:
        if result.predictions:
            principles = [
                {"name": f"Prediction {i + 1}", "description": pred}
                for i, pred in enumerate(result.predictions)
            ]
            directions = llm_client.generate_solution_directions(
                problem_text,
                improving_param="technology evolution",
                worsening_param="current limitations",
                principles=principles,
                patent_examples=patent_examples,
            )
            solution_directions = [d.model_dump() for d in directions.directions]
    except Exception:
        logger.warning("Solution direction generation failed, continuing without")

    current = result.current_stage
    return AnalysisResult(
        problem=problem_text,
        method="trends",
        ideal_final_result=ideal_final_result,
        reasoning=f"Technology is at Stage {current.get('stage', '?')} "
        f"({current.get('stage_name', '?')}) of trend "
        f"'{result.trend_name}'",
        patent_examples=patent_examples,
        solution_directions=solution_directions,
        details={
            "current_stage": result.current_stage,
            "trend_name": result.trend_name,
            "next_stages": result.next_stages,
            "predictions": result.predictions,
        },
    )
