"""Tests for Su-Field analysis pipeline."""

from unittest.mock import MagicMock

from triz_ai.engine.su_field import analyze_su_field
from triz_ai.llm.client import (
    SolutionDirection,
    SolutionDirectionBatch,
    SuFieldResult,
)


def test_analyze_su_field_returns_result():
    mock_llm = MagicMock()
    mock_llm.analyze_su_field.return_value = SuFieldResult(
        substances=["power module", "sensor"],
        field="ultrasonic",
        problem_type="incomplete",
        standard_solutions=[
            {
                "id": "1.1.1",
                "name": "Complete an incomplete Su-Field",
                "applicability": "Add missing detection field",
            }
        ],
    )
    mock_llm.generate_solution_directions.return_value = SolutionDirectionBatch(
        directions=[
            SolutionDirection(
                title="Add ultrasonic detection",
                description="Use ultrasonic field to detect delamination.",
                principles_applied=["Complete Su-Field"],
            )
        ]
    )
    mock_llm.get_embedding.return_value = [0.1] * 768

    result = analyze_su_field("detect delamination without sensors", "IFR", mock_llm)
    assert result.method == "su_field"
    assert result.details["problem_type"] == "incomplete"
    assert result.details["field"] == "ultrasonic"
    assert len(result.details["standard_solutions"]) == 1


def test_analyze_su_field_without_store():
    mock_llm = MagicMock()
    mock_llm.analyze_su_field.return_value = SuFieldResult(
        substances=["A", "B"],
        field="thermal",
        problem_type="harmful",
        standard_solutions=[],
    )
    mock_llm.generate_solution_directions.return_value = SolutionDirectionBatch(directions=[])
    mock_llm.get_embedding.return_value = [0.1] * 768

    result = analyze_su_field("test", None, mock_llm, store=None)
    assert result.patent_examples == []
