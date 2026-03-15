"""Tests for trimming analysis pipeline."""

from unittest.mock import MagicMock

from triz_ai.engine.trimming import analyze_trimming
from triz_ai.llm.client import (
    SolutionDirection,
    SolutionDirectionBatch,
    TrimmingResult,
)


def test_analyze_trimming_returns_result():
    mock_llm = MagicMock()
    mock_llm.analyze_trimming.return_value = TrimmingResult(
        components=[
            {"name": "gate driver IC", "function": "drives MOSFET gate", "cost": "high"},
            {"name": "bootstrap diode", "function": "charges bootstrap capacitor", "cost": "low"},
            {"name": "level shifter", "function": "shifts signal levels", "cost": "medium"},
        ],
        trimming_candidates=[
            {
                "component": "level shifter",
                "reason": "Gate driver IC can perform level shifting internally",
                "rule": "B",
            }
        ],
        redistributed_functions=[
            {
                "function": "level shifting",
                "from": "level shifter",
                "to": "gate driver IC",
            }
        ],
    )
    mock_llm.generate_solution_directions.return_value = SolutionDirectionBatch(
        directions=[
            SolutionDirection(
                title="Integrated gate driver",
                description="Use gate driver with built-in level shifter.",
                principles_applied=["Trim level shifter"],
            )
        ]
    )
    mock_llm.get_embedding.return_value = [0.1] * 768

    result = analyze_trimming("reduce BOM cost of gate driver circuit", "IFR", mock_llm)
    assert result.method == "trimming"
    assert len(result.details["components"]) == 3
    assert len(result.details["trimming_candidates"]) == 1
    assert result.details["trimming_candidates"][0]["rule"] == "B"


def test_analyze_trimming_no_candidates():
    mock_llm = MagicMock()
    mock_llm.analyze_trimming.return_value = TrimmingResult(
        components=[{"name": "A", "function": "essential", "cost": "low"}],
        trimming_candidates=[],
        redistributed_functions=[],
    )
    mock_llm.generate_solution_directions.return_value = SolutionDirectionBatch(directions=[])
    mock_llm.get_embedding.return_value = [0.1] * 768

    result = analyze_trimming("test", None, mock_llm)
    assert result.details["trimming_candidates"] == []
