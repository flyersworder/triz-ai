"""Tests for function analysis pipeline."""

from unittest.mock import MagicMock

from triz_ai.engine.function_analysis import analyze_functions
from triz_ai.llm.client import (
    FunctionAnalysisResult,
    SolutionDirection,
    SolutionDirectionBatch,
)


def test_analyze_functions_returns_result():
    mock_llm = MagicMock()
    mock_llm.analyze_functions.return_value = FunctionAnalysisResult(
        components=[
            {"name": "adhesive", "role": "bonds die to substrate"},
            {"name": "silicon die", "role": "active component"},
        ],
        functions=[
            {"subject": "adhesive", "action": "bonds", "object": "silicon die", "type": "useful"},
            {
                "subject": "adhesive",
                "action": "stresses",
                "object": "silicon die",
                "type": "harmful",
            },
        ],
        problem_functions=[
            {
                "subject": "adhesive",
                "action": "stresses",
                "object": "silicon die",
                "problem": "CTE mismatch causes cracking",
            }
        ],
        recommendations=["Use compliant adhesive layer", "Add stress buffer"],
    )
    mock_llm.generate_solution_directions.return_value = SolutionDirectionBatch(
        directions=[
            SolutionDirection(
                title="Compliant bonding",
                description="Replace rigid adhesive with compliant one.",
                principles_applied=["Recommendation 1"],
            )
        ]
    )
    mock_llm.get_embedding.return_value = [0.1] * 768

    result = analyze_functions("adhesive damages silicon die", "IFR", mock_llm)
    assert result.method == "function_analysis"
    assert len(result.details["components"]) == 2
    assert len(result.details["problem_functions"]) == 1
    assert len(result.details["recommendations"]) == 2


def test_analyze_functions_no_problems():
    mock_llm = MagicMock()
    mock_llm.analyze_functions.return_value = FunctionAnalysisResult(
        components=[{"name": "A", "role": "does stuff"}],
        functions=[{"subject": "A", "action": "works", "object": "B", "type": "useful"}],
        problem_functions=[],
        recommendations=[],
    )
    mock_llm.generate_solution_directions.return_value = SolutionDirectionBatch(directions=[])
    mock_llm.get_embedding.return_value = [0.1] * 768

    result = analyze_functions("test", None, mock_llm)
    assert result.details["problem_functions"] == []
