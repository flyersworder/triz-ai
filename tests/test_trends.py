"""Tests for trends analysis pipeline."""

from unittest.mock import MagicMock

from triz_ai.engine.trends import analyze_trends
from triz_ai.llm.client import (
    SolutionDirection,
    SolutionDirectionBatch,
    TrendsResult,
)


def test_analyze_trends_returns_result():
    mock_llm = MagicMock()
    mock_llm.analyze_trends.return_value = TrendsResult(
        current_stage={
            "trend_id": 4,
            "trend_name": "Transition to micro-level",
            "stage": 2,
            "stage_name": "Molecular/chemical level",
        },
        trend_name="Transition to micro-level",
        next_stages=[
            {
                "stage": 3,
                "name": "Field-based mechanism",
                "description": "SiC packaging will use field-based bonding",
            }
        ],
        predictions=[
            "Sintered silver will replace solder",
            "Direct bonded copper will evolve to active metal brazing",
        ],
    )
    mock_llm.generate_solution_directions.return_value = SolutionDirectionBatch(
        directions=[
            SolutionDirection(
                title="Sintered interconnects",
                description="Move to sintered silver for higher reliability.",
                principles_applied=["Prediction 1"],
            )
        ]
    )
    mock_llm.get_embedding.return_value = [0.1] * 768

    result = analyze_trends("next generation SiC packaging", "IFR", mock_llm)
    assert result.method == "trends"
    assert result.details["trend_name"] == "Transition to micro-level"
    assert result.details["current_stage"]["stage"] == 2
    assert len(result.details["predictions"]) == 2


def test_analyze_trends_without_store():
    mock_llm = MagicMock()
    mock_llm.analyze_trends.return_value = TrendsResult(
        current_stage={
            "trend_id": 1,
            "trend_name": "Ideality",
            "stage": 1,
            "stage_name": "Initial",
        },
        trend_name="Ideality",
        next_stages=[],
        predictions=["Will improve"],
    )
    mock_llm.generate_solution_directions.return_value = SolutionDirectionBatch(directions=[])
    mock_llm.get_embedding.return_value = [0.1] * 768

    result = analyze_trends("test", None, mock_llm, store=None)
    assert result.patent_examples == []
