"""Tests for physical contradiction pipeline."""

from unittest.mock import MagicMock

from triz_ai.engine.physical import analyze_physical
from triz_ai.llm.client import (
    PhysicalContradictionResult,
    SolutionDirection,
    SolutionDirectionBatch,
)


def test_analyze_physical_returns_result():
    mock_llm = MagicMock()
    mock_llm.extract_physical_contradiction.return_value = PhysicalContradictionResult(
        property="stiffness",
        requirement_a="rigid for structural support",
        requirement_b="flexible for thermal cycling",
        separation_type="separation_in_time",
        separation_principles=[
            {"id": 1, "name": "Separation in Time", "technique": "Alternate between states"}
        ],
    )
    mock_llm.generate_solution_directions.return_value = SolutionDirectionBatch(
        directions=[
            SolutionDirection(
                title="Time-based stiffness control",
                description="Use SMA to switch stiffness modes.",
                principles_applied=["Separation in Time"],
            )
        ]
    )
    mock_llm.get_embedding.return_value = [0.1] * 768

    result = analyze_physical("solder joint must be rigid AND flexible", "IFR", mock_llm)
    assert result.method == "physical_contradiction"
    assert result.details["property"] == "stiffness"
    assert result.details["separation_type"] == "separation_in_time"
    assert len(result.details["separation_principles"]) == 1
    assert result.ideal_final_result == "IFR"


def test_analyze_physical_without_store():
    mock_llm = MagicMock()
    mock_llm.extract_physical_contradiction.return_value = PhysicalContradictionResult(
        property="temperature",
        requirement_a="hot",
        requirement_b="cold",
        separation_type="separation_in_space",
        separation_principles=[],
    )
    mock_llm.generate_solution_directions.return_value = SolutionDirectionBatch(directions=[])
    mock_llm.get_embedding.return_value = [0.1] * 768

    result = analyze_physical("must be hot and cold", None, mock_llm, store=None)
    assert result.patent_examples == []
    assert result.method == "physical_contradiction"
