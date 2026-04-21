"""ARIZ-85C deep analysis models and orchestrator."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from pydantic import BaseModel

from triz_ai.engine.analyzer import AnalysisResult
from triz_ai.llm.client import TrizAIError

if TYPE_CHECKING:
    from triz_ai.llm.client import LLMClient
    from triz_ai.patents.repository import PatentRepository

logger = logging.getLogger(__name__)


class TechnicalContradiction(BaseModel):
    improving_param_id: int
    improving_param_name: str
    worsening_param_id: int
    worsening_param_name: str
    intensified_description: str


class PhysicalContradictionModel(BaseModel):
    property: str
    macro_requirement: str
    micro_requirement: str


class ResourceInventory(BaseModel):
    substances: list[str]
    fields: list[str]
    time_resources: list[str]
    space_resources: list[str]


class StructuredProblemModel(BaseModel):
    original_problem: str
    reformulated_problem: str
    technical_contradiction_1: TechnicalContradiction
    technical_contradiction_2: TechnicalContradiction
    physical_contradiction: PhysicalContradictionModel | None = None
    ideal_final_result: str
    resource_inventory: ResourceInventory
    recommended_tools: list[str]
    recommended_research_tools: list[str] = []
    reasoning: str


class VerifiedCandidate(BaseModel):
    method: str
    satisfies_ifr: bool
    ifr_gap: str
    ideality_score: float
    key_insight: str


class SynthesizedSolution(BaseModel):
    title: str
    description: str
    principles_applied: list[str]
    supersystem_changes: list[str]
    ideality_score: float


class SolutionVerification(BaseModel):
    verified_candidates: list[VerifiedCandidate]
    any_satisfies_ifr: bool
    synthesized_solutions: list[SynthesizedSolution]
    reasoning: str


class DeepAnalysisResult(BaseModel):
    problem_model: StructuredProblemModel
    tool_results: list[AnalysisResult]
    tools_used: list[str]
    verification: SolutionVerification
    used_escape_hatch: bool = False


# --- Orchestration logic ---

VALID_TOOLS = {
    "technical_contradiction",
    "physical_contradiction",
    "su_field",
    "function_analysis",
    "trimming",
    "trends",
}


def _select_tools(problem_model: StructuredProblemModel) -> list[str]:
    """Select 2-4 tools based on LLM recommendations.

    - Always includes technical_contradiction
    - Validates against known methods, drops invalid
    - Clamps to 2-4 tools
    """
    tools: list[str] = []
    for t in problem_model.recommended_tools:
        normalized = t.strip().lower().replace("-", "_")
        if normalized in VALID_TOOLS and normalized not in tools:
            tools.append(normalized)

    # Always include technical_contradiction
    if "technical_contradiction" not in tools:
        tools.insert(0, "technical_contradiction")

    # Clamp to 2-4
    if len(tools) < 2:
        # Add physical_contradiction as second if only 1
        for fallback in ["physical_contradiction", "su_field", "trimming"]:
            if fallback not in tools:
                tools.append(fallback)
                break
    tools = tools[:4]

    return tools


def _get_pipeline_fn(method: str):
    """Import and return the pipeline function for a given method."""
    from triz_ai.engine.analyzer import analyze_contradiction
    from triz_ai.engine.function_analysis import analyze_functions
    from triz_ai.engine.physical import analyze_physical
    from triz_ai.engine.su_field import analyze_su_field
    from triz_ai.engine.trends import analyze_trends
    from triz_ai.engine.trimming import analyze_trimming

    pipelines = {
        "technical_contradiction": analyze_contradiction,
        "physical_contradiction": analyze_physical,
        "su_field": analyze_su_field,
        "function_analysis": analyze_functions,
        "trimming": analyze_trimming,
        "trends": analyze_trends,
    }
    return pipelines[method]


def _select_research_tools(research_tools, problem_model):
    """Select research tools based on LLM recommendations.

    If LLM recommends specific tools, use those. Otherwise use all available.
    """
    if not research_tools:
        return None
    recommended = problem_model.recommended_research_tools
    if recommended:
        available = {t.name: t for t in research_tools}
        selected = [available[name] for name in recommended if name in available]
        if selected:
            return selected
    return research_tools  # fallback: use all


def _run_tools(
    tools: list[str],
    problem_text: str,
    ifr: str,
    llm_client: LLMClient,
    store: PatentRepository | None,
    research_tools: list | None = None,
) -> list[AnalysisResult]:
    """Run multiple TRIZ pipelines in parallel using ThreadPoolExecutor."""
    results: list[AnalysisResult] = []

    def _run_one(method: str) -> AnalysisResult:
        pipeline_fn = _get_pipeline_fn(method)
        return pipeline_fn(problem_text, ifr, llm_client, store, research_tools=research_tools)

    with ThreadPoolExecutor(max_workers=len(tools)) as executor:
        future_to_method = {executor.submit(_run_one, m): m for m in tools}
        for future in as_completed(future_to_method):
            method = future_to_method[future]
            try:
                results.append(future.result())
            except Exception:
                logger.warning(
                    "Pipeline '%s' failed in deep mode, skipping", method, exc_info=True
                )

    if not results:
        raise TrizAIError(
            "All TRIZ pipelines failed in deep mode. Check your LLM configuration and API key."
        )

    return results


def orchestrate_deep(
    problem_text: str,
    llm_client: LLMClient,
    store: PatentRepository | None,
    deep_model: str | None = None,
    reasoning_effort: str | None = None,
    research_tools: list | None = None,
) -> DeepAnalysisResult:
    """Run the full ARIZ-85C deep analysis: 3 passes with escape hatch.

    Context-stage research tools run first to enrich problem_text.

    Pass 1: Deep reformulation → StructuredProblemModel  (deep_model)
    Pass 2: Multi-tool research (parallel)               (base model)
    Pass 3: Verify + synthesize → SolutionVerification   (deep_model)
    Escape hatch: If no candidate satisfies IFR, swap TC1↔TC2 and retry once.

    Args:
        deep_model: Optional model for Passes 1 & 3 (e.g. a reasoning model).
            Falls back to llm_client's default model if not set.
        reasoning_effort: Optional reasoning effort (low/medium/high) for
            Passes 1 & 3. Passed to litellm which translates across providers.
        research_tools: Optional list of ResearchTool instances. Context-stage
            tools enrich problem_text before Pass 1; search/enrichment-stage
            tools run within each pipeline in Pass 2.
    """
    # Run context tools before Pass 1
    if research_tools:
        from triz_ai.tools import run_stage_tools

        context_results = run_stage_tools(research_tools, "context", problem_text)
        if context_results:
            context_parts = [r.get("content", "") for r in context_results if r.get("content")]
            if context_parts:
                additional_context = "\n\n".join(context_parts)
                problem_text = (
                    f"Additional context:\n{additional_context}\n\nProblem: {problem_text}"
                )

    # Build research tool descriptions for the prompt
    research_tool_descriptions = None
    if research_tools:
        research_tool_descriptions = [
            {"name": t.name, "description": t.description, "stages": t.stages}
            for t in research_tools
        ]

    # Pass 1: Deep reformulation (uses deep_model if provided)
    problem_model = llm_client.deep_reformulate(
        problem_text,
        model=deep_model,
        reasoning_effort=reasoning_effort,
        research_tool_descriptions=research_tool_descriptions,
    )

    # Tool selection
    tools = _select_tools(problem_model)

    # Select research tools based on LLM recommendation
    selected_research_tools = _select_research_tools(research_tools, problem_model)

    # Pass 2: Multi-tool research (parallel, uses base model)
    tool_results = _run_tools(
        tools,
        problem_model.reformulated_problem,
        problem_model.ideal_final_result,
        llm_client,
        store,
        research_tools=selected_research_tools,
    )

    # Pass 3: Verify + synthesize (uses deep_model if provided)
    candidates = [r.model_dump() for r in tool_results]
    verification = llm_client.verify_and_synthesize(
        problem_model,
        candidates,
        model=deep_model,
        reasoning_effort=reasoning_effort,
    )

    # Escape hatch: if no candidate satisfies IFR, swap TCs and retry once
    used_escape_hatch = False
    if not verification.any_satisfies_ifr:
        logger.info("No candidate satisfies IFR — activating escape hatch (TC swap)")
        used_escape_hatch = True

        # Swap TC1 ↔ TC2 in the problem model
        swapped_model = problem_model.model_copy(
            update={
                "technical_contradiction_1": problem_model.technical_contradiction_2,
                "technical_contradiction_2": problem_model.technical_contradiction_1,
            }
        )

        # Re-run Pass 2 with swapped TCs (same reformulated problem + IFR)
        tool_results = _run_tools(
            tools,
            swapped_model.reformulated_problem,
            swapped_model.ideal_final_result,
            llm_client,
            store,
            research_tools=selected_research_tools,
        )

        # Re-run Pass 3
        candidates = [r.model_dump() for r in tool_results]
        verification = llm_client.verify_and_synthesize(
            swapped_model,
            candidates,
            model=deep_model,
            reasoning_effort=reasoning_effort,
        )
        problem_model = swapped_model

    # Self-evolution: collect web search observations from all tool results
    if store is not None and research_tools:
        try:
            from triz_ai.evolution.self_evolve import (
                collect_search_observations,
                maybe_auto_consolidate,
            )

            total_collected = sum(
                collect_search_observations(tool_result, store) for tool_result in tool_results
            )
            if total_collected > 0:
                store.increment_analysis_count()
            maybe_auto_consolidate(llm_client, store)
        except Exception:
            logger.warning("Self-evolution collection failed, continuing", exc_info=True)

    return DeepAnalysisResult(
        problem_model=problem_model,
        tool_results=tool_results,
        tools_used=tools,
        verification=verification,
        used_escape_hatch=used_escape_hatch,
    )
