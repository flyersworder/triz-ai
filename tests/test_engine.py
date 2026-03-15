"""Tests for engine modules with mocked LLM."""

from unittest.mock import MagicMock

import pytest

from triz_ai.engine.analyzer import analyze
from triz_ai.engine.classifier import classify
from triz_ai.engine.evaluator import evaluate
from triz_ai.engine.generator import discover
from triz_ai.llm.client import (
    ExtractedContradiction,
    Idea,
    IdeaBatch,
    PatentClassification,
    SolutionDirection,
    SolutionDirectionBatch,
)
from triz_ai.patents.store import Classification, Patent, PatentStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    s = PatentStore(db_path=db_path)
    s.init_db()
    yield s
    s.close()


@pytest.fixture
def mock_llm():
    client = MagicMock()
    client.extract_contradiction.return_value = ExtractedContradiction(
        improving_param=9,
        worsening_param=1,
        reasoning="Speed vs weight tradeoff",
    )
    client.classify_patent.return_value = PatentClassification(
        principle_ids=[1, 14, 35],
        contradiction={"improving": 9, "worsening": 1},
        confidence=0.85,
        reasoning="Uses segmentation and curvature",
    )
    client.generate_ideas.return_value = IdeaBatch(
        ideas=[
            Idea(
                idea="Use segmented battery cells",
                principle_id=1,
                reasoning="Segmentation improves thermal management",
                source_patent_id="US123456",
            )
        ]
    )
    client.generate_solution_directions.return_value = SolutionDirectionBatch(
        directions=[
            SolutionDirection(
                title="Segmented lightweight frame",
                description="Apply segmentation to the car frame to reduce weight.",
                principles_applied=["Segmentation", "Spheroidality — Curvature"],
            )
        ]
    )
    client.get_embedding.return_value = [0.1] * 768
    return client


class TestAnalyzer:
    def test_analyze_returns_result(self, mock_llm, store):
        result = analyze("Make a lighter faster car", llm_client=mock_llm, store=store)
        assert result.problem == "Make a lighter faster car"
        assert result.improving_param["id"] == 9
        assert result.worsening_param["id"] == 1
        assert len(result.recommended_principles) > 0
        mock_llm.extract_contradiction.assert_called_once()

    def test_analyze_without_store(self, mock_llm):
        result = analyze("Make a lighter faster car", llm_client=mock_llm, store=None)
        assert result.patent_examples == []

    def test_analyze_solution_directions(self, mock_llm, store):
        result = analyze("Make a lighter faster car", llm_client=mock_llm, store=store)
        assert len(result.solution_directions) == 1
        assert result.solution_directions[0]["title"] == "Segmented lightweight frame"
        assert "Segmentation" in result.solution_directions[0]["principles_applied"]
        mock_llm.generate_solution_directions.assert_called_once()

    def test_analyze_enriched_patents(self, mock_llm, store):
        """Patent examples should include assignee, filing_date, matched_principles."""
        # Add a patent with classification and embedding
        store.insert_patent(
            Patent(
                id="P1",
                title="Lightweight car frame",
                assignee="Tesla Inc",
                filing_date="2024-06-01",
            ),
            embedding=[0.1] * 768,
        )
        store.insert_classification(
            Classification(
                patent_id="P1",
                principle_ids=[1, 14],
                contradiction={"improving": 9, "worsening": 1},
                confidence=0.9,
            )
        )
        result = analyze("Make a lighter faster car", llm_client=mock_llm, store=store)
        if result.patent_examples:
            pe = result.patent_examples[0]
            assert "assignee" in pe
            assert "filing_date" in pe
            assert "matched_principles" in pe


class TestClassifier:
    def test_classify_patent_text(self, mock_llm):
        result = classify("A battery with segmented cells...", llm_client=mock_llm)
        assert 1 in result.principle_ids
        assert result.confidence == 0.85

    def test_classify_and_store(self, mock_llm, store):
        store.insert_patent(Patent(id="P1", title="Test"))
        classify(
            "A battery with segmented cells...",
            patent_id="P1",
            llm_client=mock_llm,
            store=store,
        )
        stored = store.get_classification("P1")
        assert stored is not None
        assert stored.principle_ids == [1, 14, 35]

    def test_classify_auto_records_observations(self, mock_llm, store):
        """Classification should auto-record matrix observations."""
        store.insert_patent(Patent(id="P2", title="Test Patent"))
        classify(
            "A battery with segmented cells...",
            patent_id="P2",
            llm_client=mock_llm,
            store=store,
        )
        # Check that observations were recorded for each principle
        # mock_llm returns principle_ids=[1, 14, 35] with contradiction improving=9, worsening=1
        obs = store.get_matrix_observations(min_count=1)
        assert (9, 1) in obs
        principle_ids_in_obs = [pid for pid, _, _ in obs[(9, 1)]]
        assert 1 in principle_ids_in_obs
        assert 14 in principle_ids_in_obs
        assert 35 in principle_ids_in_obs


class TestGenerator:
    def test_discover_with_data(self, mock_llm, store):
        # Add classified patents
        for i in range(3):
            store.insert_patent(Patent(id=f"P{i}", title=f"Patent {i}", domain="battery"))
            store.insert_classification(
                Classification(
                    patent_id=f"P{i}",
                    principle_ids=[1, 2],
                    contradiction={"improving": 9, "worsening": 1},
                    confidence=0.9,
                )
            )
        result = discover("battery", llm_client=mock_llm, store=store)
        assert result.domain == "battery"
        assert result.total_patents == 3
        assert len(result.principle_usage) == 40
        assert len(result.underused_principles) > 0

    def test_discover_empty_domain(self, mock_llm, store):
        result = discover("nonexistent", llm_client=mock_llm, store=store)
        assert result.total_patents == 0
        assert len(result.underused_principles) == 40  # all underused


class TestCLICommands:
    def test_classify_not_a_cli_command(self):
        """The classify command should no longer exist in the CLI."""
        from triz_ai.cli import app

        command_names = [cmd.name for cmd in app.registered_commands]
        assert "classify" not in command_names


class TestEvaluator:
    def test_evaluate_with_no_store(self, mock_llm, store):
        result = evaluate("A new battery design", "battery", llm_client=mock_llm, store=store)
        assert result.novelty_score >= 0
        assert result.novelty_score <= 1
        assert result.domain == "battery"
