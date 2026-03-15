"""LLM client wrapping litellm for TRIZ operations."""

import json
import logging
from typing import TypeVar

import litellm
from pydantic import BaseModel

from triz_ai.config import load_config
from triz_ai.llm.prompts import (
    classify_patent_prompt,
    classify_problem_prompt,
    cluster_patents_prompt,
    deep_reformulation_prompt,
    extract_contradiction_prompt,
    extract_physical_contradiction_prompt,
    function_analysis_prompt,
    generate_ideas_prompt,
    ideal_final_result_prompt,
    propose_candidate_parameter_prompt,
    propose_candidate_principle_prompt,
    root_cause_analysis_prompt,
    seed_matrix_prompt,
    solution_directions_prompt,
    solution_verification_prompt,
    su_field_analysis_prompt,
    trends_analysis_prompt,
    trimming_analysis_prompt,
)

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

# Suppress litellm's noisy console output
litellm.suppress_debug_info = True  # type: ignore[assignment]
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class TrizAIError(Exception):
    """User-facing error with actionable guidance."""


# Response models
class ExtractedContradiction(BaseModel):
    improving_param: int
    worsening_param: int
    reasoning: str
    confidence: float = 1.0


class PatentClassification(BaseModel):
    principle_ids: list[int]
    contradiction: dict  # {"improving": int, "worsening": int}
    confidence: float
    reasoning: str


class Idea(BaseModel):
    idea: str
    principle_id: int
    reasoning: str
    source_patent_id: str | None = None


class IdeaBatch(BaseModel):
    ideas: list[Idea]


class SolutionDirection(BaseModel):
    title: str
    description: str
    principles_applied: list[str]


class SolutionDirectionBatch(BaseModel):
    directions: list[SolutionDirection]


class CandidatePrincipleProposal(BaseModel):
    name: str
    description: str
    how_it_differs: str
    confidence: float


class CandidateParameterProposal(BaseModel):
    name: str
    description: str
    how_it_differs: str
    confidence: float


class MatrixEntry(BaseModel):
    improving: int
    worsening: int
    principles: list[int]


class MatrixSeedResult(BaseModel):
    entries: list[MatrixEntry]


# --- Multi-tool routing models ---


class ProblemClassification(BaseModel):
    primary_method: str  # one of the 6 TRIZ methods
    secondary_method: str | None = None
    reasoning: str
    confidence: float
    reformulated_problem: str


class IdealFinalResult(BaseModel):
    ideal_result: str
    reasoning: str


class RootCauseAnalysis(BaseModel):
    root_causes: list[str]
    reformulated_problem: str
    reasoning: str


class PhysicalContradictionResult(BaseModel):
    property: str
    requirement_a: str
    requirement_b: str
    separation_type: str  # one of the 4 separation categories
    separation_principles: list[dict]  # [{"id": int, "name": str, "technique": str}]


class SuFieldResult(BaseModel):
    substances: list[str]
    field: str
    problem_type: str  # incomplete | harmful | inefficient
    standard_solutions: list[dict]  # [{"id": str, "name": str, "applicability": str}]


class FunctionAnalysisResult(BaseModel):
    components: list[dict]  # [{"name": str, "role": str}]
    functions: list[dict]  # [{"subject": str, "action": str, "object": str, "type": str}]
    problem_functions: list[dict]  # [{subject, action, object, problem}]
    recommendations: list[str]


class TrimmingResult(BaseModel):
    components: list[dict]  # [{"name": str, "function": str, "cost": str}]
    trimming_candidates: list[dict]  # [{"component": str, "reason": str, "rule": str}]
    redistributed_functions: list[dict]  # [{"function": str, "from": str, "to": str}]


class TrendsResult(BaseModel):
    current_stage: dict  # {"trend_id": int, "trend_name": str, "stage": int, "stage_name": str}
    trend_name: str
    next_stages: list[dict]  # [{"stage": int, "name": str, "description": str}]
    predictions: list[str]


def _friendly_error(e: Exception) -> TrizAIError:
    """Convert litellm exceptions to user-friendly error messages."""
    error_str = str(e)
    error_type = type(e).__name__

    if "AuthenticationError" in error_type or "401" in error_str:
        return TrizAIError(
            "Authentication failed. Check your API key:\n"
            "  1. Add OPENROUTER_API_KEY=your-key to .env file\n"
            "  2. Or set api_key in ~/.triz-ai/config.yaml\n"
            "  3. Or export OPENROUTER_API_KEY=your-key"
        )
    if "RateLimitError" in error_type or "429" in error_str:
        return TrizAIError(
            "Rate limit exceeded. Wait a moment and try again, "
            "or switch to a different model with --model."
        )
    if "Timeout" in error_type:
        return TrizAIError(
            "Request timed out. The LLM provider may be slow or unreachable. "
            "Check your network or try a different model with --model."
        )
    if "APIConnectionError" in error_type or "Connection" in error_str:
        return TrizAIError(
            "Cannot connect to LLM provider. Check your network connection.\n"
            "If using Ollama, make sure it's running: ollama serve"
        )
    if "NotFoundError" in error_type or "404" in error_str:
        return TrizAIError(
            "Model not found. Check your model name in config.\n"
            "  Current model may not be available on your provider."
        )
    return TrizAIError(f"LLM request failed: {e}")


def _is_retryable(e: Exception) -> bool:
    """Check if an error is worth retrying with a stricter prompt.

    Only retry on validation/parsing errors, not auth/network/rate issues.
    """
    error_type = type(e).__name__
    non_retryable = {
        "AuthenticationError",
        "RateLimitError",
        "APIConnectionError",
        "Timeout",
        "NotFoundError",
        "BudgetExceededError",
    }
    return not any(name in error_type for name in non_retryable)


class LLMClient:
    def __init__(self, model: str | None = None, classify_model: str | None = None):
        config = load_config()
        self.model = model or config.llm.default_model
        self.classify_model = classify_model or config.llm.classify_model
        self.api_base = config.llm.api_base
        self.api_key = config.llm.api_key
        self.embedding_model = config.embeddings.model
        self.embedding_dimensions = config.embeddings.dimensions
        self.embedding_api_base = config.embeddings.api_base
        self.embedding_api_key = config.embeddings.api_key

    def _completion_kwargs(self) -> dict:
        """Build optional kwargs for litellm.completion."""
        kwargs: dict = {}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return kwargs

    def _embedding_kwargs(self) -> dict:
        """Build optional kwargs for litellm.embedding."""
        kwargs: dict = {}
        if self.embedding_api_base:
            kwargs["api_base"] = self.embedding_api_base
        if self.embedding_api_key:
            kwargs["api_key"] = self.embedding_api_key
        return kwargs

    def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        retry: bool = True,
        model: str | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> T:
        """Call LLM and validate response against pydantic model.

        On malformed response, retry once with stricter prompt, then fail.
        Auth/network errors are raised immediately without retry.

        Args:
            model: Optional model override (defaults to self.model).
            max_tokens: Optional max output tokens (useful for structured
                responses to avoid reserving large output windows).
            reasoning_effort: Optional reasoning effort level for reasoning
                models (low/medium/high). Passed to litellm which translates
                across providers (Anthropic, OpenAI o-series, DeepSeek, etc.).
        """
        use_model = model or self.model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs = self._completion_kwargs()
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        try:
            response = litellm.completion(
                model=use_model,
                messages=messages,
                response_format={"type": "json_object"},
                **kwargs,
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)
            return response_model.model_validate(data)
        except Exception as e:
            if retry and _is_retryable(e):
                logger.debug("First attempt failed (%s), retrying with stricter prompt", e)
                strict_system = (
                    system_prompt
                    + "\n\nIMPORTANT: You MUST respond with valid JSON matching this exact"
                    f" schema: {response_model.model_json_schema()}"
                )
                return self._complete(
                    strict_system,
                    user_prompt,
                    response_model,
                    retry=False,
                    model=use_model,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                )
            raise _friendly_error(e) from e

    def extract_contradiction(self, problem_text: str) -> ExtractedContradiction:
        """Extract technical contradiction from problem description."""
        return self._complete(
            extract_contradiction_prompt(),
            problem_text,
            ExtractedContradiction,
        )

    def classify_patent(self, patent_text: str) -> PatentClassification:
        """Classify a patent by TRIZ principles.

        Uses the classify_model (smaller/cheaper) instead of the default model.
        max_tokens=1024 avoids reserving large output windows on pay-per-token models.
        """
        return self._complete(
            classify_patent_prompt(),
            patent_text,
            PatentClassification,
            model=self.classify_model,
            max_tokens=1024,
        )

    def generate_solution_directions(
        self,
        problem_text: str,
        improving_param: str,
        worsening_param: str,
        principles: list[dict],
        patent_examples: list[dict],
    ) -> SolutionDirectionBatch:
        """Generate concrete solution directions from TRIZ analysis."""
        principles_text = "\n".join(f"- {p['name']}: {p['description']}" for p in principles)
        patents_text = ""
        if patent_examples:
            patent_lines = []
            for pe in patent_examples[:5]:
                assignee = pe.get("assignee") or "Unknown"
                patent_lines.append(f"- {pe['title']} ({assignee})")
            patents_text = "\n\nRelated patents:\n" + "\n".join(patent_lines)

        user_prompt = (
            f"Problem: {problem_text}\n\n"
            f"Contradiction: Improving '{improving_param}' worsens '{worsening_param}'\n\n"
            f"Recommended TRIZ principles:\n{principles_text}"
            f"{patents_text}\n\n"
            "Generate 2-3 concrete solution directions."
        )
        return self._complete(solution_directions_prompt(), user_prompt, SolutionDirectionBatch)

    def generate_ideas(
        self,
        domain: str,
        underused_principles: list[dict],
        existing_patents: list[str],
    ) -> IdeaBatch:
        """Generate novel ideas using underused principles."""
        principles_text = "\n".join(
            f"- Principle {p['id']}: {p['name']} — {p['description']}"
            for p in underused_principles
        )
        patents_text = "\n".join(f"- {p}" for p in existing_patents[:10])

        user_prompt = (
            f"Domain: {domain}\n\n"
            f"Underused TRIZ principles in this domain:\n{principles_text}\n\n"
            f"Existing patents in this domain:\n{patents_text}\n\n"
            f"Generate 3-5 novel ideas applying these underused principles to {domain}."
        )
        return self._complete(generate_ideas_prompt(), user_prompt, IdeaBatch)

    def propose_candidate_parameter(self, patent_cluster: list[str]) -> CandidateParameterProposal:
        """Propose a candidate new engineering parameter from a cluster of patents."""
        patents_text = "\n---\n".join(patent_cluster)
        return self._complete(
            propose_candidate_parameter_prompt(),
            patents_text,
            CandidateParameterProposal,
        )

    def propose_candidate_principle(self, patent_cluster: list[str]) -> CandidatePrincipleProposal:
        """Propose a candidate new TRIZ principle from a cluster of patents."""
        patents_text = "\n---\n".join(patent_cluster)
        return self._complete(
            propose_candidate_principle_prompt(),
            patents_text,
            CandidatePrincipleProposal,
        )

    def seed_matrix_row(self, improving: int, worsening_params: list[int]) -> MatrixSeedResult:
        """Seed missing matrix cells for an improving parameter via LLM."""
        from triz_ai.knowledge.contradictions import load_matrix
        from triz_ai.knowledge.parameters import get_parameter

        imp_param = get_parameter(improving)
        imp_name = imp_param.name if imp_param else f"Parameter {improving}"

        wp_dicts = []
        for wid in worsening_params:
            wp = get_parameter(wid)
            wp_dicts.append({"id": wid, "name": wp.name if wp else f"Parameter {wid}"})

        # Build 3 example rows from existing matrix
        matrix = load_matrix()
        example_rows: list[str] = []
        for (imp_id, wor_id), principles in matrix.items():
            if len(example_rows) >= 3:
                break
            example_rows.append(
                f'{{"improving": {imp_id}, "worsening": {wor_id}, "principles": {principles}}}'
            )

        prompt = seed_matrix_prompt(improving, imp_name, wp_dicts, example_rows)
        user_msg = f"Fill matrix row for improving parameter {improving}: {imp_name}"
        return self._complete(prompt, user_msg, MatrixSeedResult)

    def classify_problem(
        self, problem_text: str, model: str | None = None
    ) -> "ProblemClassification":
        """Classify a problem into the appropriate TRIZ analysis method."""
        use_model = model or self.classify_model
        return self._complete(
            classify_problem_prompt(),
            problem_text,
            ProblemClassification,
            model=use_model,
            max_tokens=1024,
        )

    def formulate_ifr(self, problem_text: str) -> "IdealFinalResult":
        """Formulate the Ideal Final Result for a problem."""
        return self._complete(
            ideal_final_result_prompt(),
            problem_text,
            IdealFinalResult,
        )

    def analyze_root_cause(self, problem_text: str) -> "RootCauseAnalysis":
        """Trace a vague problem to its root cause and reformulate."""
        return self._complete(
            root_cause_analysis_prompt(),
            problem_text,
            RootCauseAnalysis,
        )

    def extract_physical_contradiction(self, problem_text: str) -> "PhysicalContradictionResult":
        """Extract physical contradiction and recommend separation principles."""
        return self._complete(
            extract_physical_contradiction_prompt(),
            problem_text,
            PhysicalContradictionResult,
        )

    def analyze_su_field(self, problem_text: str) -> "SuFieldResult":
        """Analyze a problem using Su-Field modeling."""
        return self._complete(
            su_field_analysis_prompt(),
            problem_text,
            SuFieldResult,
        )

    def analyze_functions(self, problem_text: str) -> "FunctionAnalysisResult":
        """Perform function analysis on a system."""
        return self._complete(
            function_analysis_prompt(),
            problem_text,
            FunctionAnalysisResult,
        )

    def analyze_trimming(self, problem_text: str) -> "TrimmingResult":
        """Analyze a system for trimming opportunities."""
        return self._complete(
            trimming_analysis_prompt(),
            problem_text,
            TrimmingResult,
        )

    def analyze_trends(self, problem_text: str) -> "TrendsResult":
        """Analyze technology evolution trends for a system."""
        return self._complete(
            trends_analysis_prompt(),
            problem_text,
            TrendsResult,
        )

    def deep_reformulate(
        self,
        problem_text: str,
        model: str | None = None,
        reasoning_effort: str | None = None,
        research_tool_descriptions: list[dict] | None = None,
    ):
        """Deep ARIZ-85C reformulation (Pass 1).

        Args:
            model: Optional model override for this pass (e.g. a reasoning model).
            reasoning_effort: Optional effort level (low/medium/high).
            research_tool_descriptions: Optional list of {"name", "description"} dicts
                for available research tools. Included in prompt so LLM can recommend
                which to use.
        """
        from triz_ai.engine.ariz import StructuredProblemModel

        return self._complete(
            deep_reformulation_prompt(research_tool_descriptions=research_tool_descriptions),
            problem_text,
            StructuredProblemModel,
            model=model,
            max_tokens=4096,
            reasoning_effort=reasoning_effort,
        )

    def verify_and_synthesize(
        self,
        problem_model,
        candidates: list[dict],
        model: str | None = None,
        reasoning_effort: str | None = None,
    ):
        """Verify candidates against IFR and synthesize (Pass 3).

        Args:
            model: Optional model override for this pass (e.g. a reasoning model).
            reasoning_effort: Optional effort level (low/medium/high).
        """
        from triz_ai.engine.ariz import SolutionVerification

        # Build compact user prompt from problem model and candidates
        candidate_summaries = []
        for c in candidates:
            dirs = c.get("solution_directions", [])
            dir_titles = ", ".join(d.get("title", "") for d in dirs)
            summary = (
                f"Method: {c['method']}\n"
                f"Reasoning: {c.get('reasoning', 'N/A')}\n"
                f"Solution directions: {dir_titles}"
            )
            candidate_summaries.append(summary)

        tc1_desc = problem_model.technical_contradiction_1.intensified_description
        tc2_desc = problem_model.technical_contradiction_2.intensified_description
        user_prompt = (
            f"Problem: {problem_model.reformulated_problem}\n\n"
            f"Ideal Final Result: {problem_model.ideal_final_result}\n\n"
            f"Technical Contradiction 1: {tc1_desc}\n"
            f"Technical Contradiction 2: {tc2_desc}\n\n"
            f"Candidates:\n\n" + "\n---\n".join(candidate_summaries)
        )

        return self._complete(
            solution_verification_prompt(),
            user_prompt,
            SolutionVerification,
            model=model,
            max_tokens=4096,
            reasoning_effort=reasoning_effort,
        )

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text."""
        try:
            response = litellm.embedding(
                model=self.embedding_model,
                input=[text],
                dimensions=self.embedding_dimensions,
                **self._embedding_kwargs(),
            )
            return response.data[0]["embedding"]
        except Exception as e:
            raise _friendly_error(e) from e

    def cluster_patents(self, patent_abstracts: list[str]) -> list[list[int]]:
        """Use LLM to semantically cluster patent abstracts.

        Returns list of clusters, each cluster is a list of indices into patent_abstracts.
        """

        class ClusterResult(BaseModel):
            clusters: list[list[int]]
            cluster_descriptions: list[str]

        abstracts_text = "\n".join(
            f"[{i}] {abstract}" for i, abstract in enumerate(patent_abstracts)
        )

        result = self._complete(cluster_patents_prompt(), abstracts_text, ClusterResult)
        return result.clusters
