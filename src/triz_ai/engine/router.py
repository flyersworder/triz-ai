"""Problem classifier + IFR + RCA + dispatch to appropriate TRIZ pipeline."""

import logging

from triz_ai.engine.analyzer import AnalysisResult, analyze_contradiction
from triz_ai.engine.function_analysis import analyze_functions
from triz_ai.engine.physical import analyze_physical
from triz_ai.engine.su_field import analyze_su_field
from triz_ai.engine.trends import analyze_trends
from triz_ai.engine.trimming import analyze_trimming
from triz_ai.llm.client import LLMClient
from triz_ai.patents.store import PatentStore

logger = logging.getLogger(__name__)

VALID_METHODS = {
    "technical_contradiction",
    "physical_contradiction",
    "su_field",
    "function_analysis",
    "trimming",
    "trends",
}

# CLI-friendly aliases (allow hyphens)
_METHOD_ALIASES = {
    "technical-contradiction": "technical_contradiction",
    "physical-contradiction": "physical_contradiction",
    "su-field": "su_field",
    "function-analysis": "function_analysis",
}


def _get_pipeline(method: str):
    """Get pipeline function by method name (lazy lookup for testability)."""
    pipelines = {
        "technical_contradiction": analyze_contradiction,
        "physical_contradiction": analyze_physical,
        "su_field": analyze_su_field,
        "function_analysis": analyze_functions,
        "trimming": analyze_trimming,
        "trends": analyze_trends,
    }
    return pipelines[method]


def _normalize_method(method: str) -> str:
    """Normalize method name, supporting both underscore and hyphen forms."""
    method = method.strip().lower()
    return _METHOD_ALIASES.get(method, method)


def route(
    problem_text: str,
    llm_client: LLMClient,
    store: PatentStore | None = None,
    method: str | None = None,
    router_model: str | None = None,
    research_tools: list | None = None,
) -> AnalysisResult:
    """Route a problem to the appropriate TRIZ analysis pipeline.

    1. Run context-stage research tools (enriches problem text for all downstream calls)
    2. Formulate IFR (always)
    3. If --method specified, skip classification
    4. Otherwise classify → get primary/secondary method
    5. If confidence < 0.4, run RCA to reformulate, then re-classify
    6. Dispatch to appropriate pipeline
    7. Attach secondary method suggestion
    """
    # Step 1: Run context tools (enriches problem text for all downstream calls)
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

    # Step 2: Formulate IFR (always, using enriched problem text)
    ideal_final_result = None
    try:
        ifr = llm_client.formulate_ifr(problem_text)
        ideal_final_result = ifr.ideal_result
    except Exception:
        logger.warning("IFR formulation failed, continuing without")

    # Step 3: Determine method
    primary_method: str
    secondary_method: str | None = None
    method_confidence: float = 1.0

    if method:
        primary_method = _normalize_method(method)
        if primary_method not in VALID_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Valid methods: {', '.join(sorted(VALID_METHODS))}"
            )
    else:
        # Step 3: Classify
        classification = llm_client.classify_problem(problem_text, model=router_model)
        primary_method = classification.primary_method
        secondary_method = classification.secondary_method
        method_confidence = classification.confidence

        # Step 4: Low confidence → RCA → re-classify
        if method_confidence < 0.4:
            try:
                rca = llm_client.analyze_root_cause(problem_text)
                problem_text = rca.reformulated_problem
                classification = llm_client.classify_problem(problem_text, model=router_model)
                primary_method = classification.primary_method
                secondary_method = classification.secondary_method
                method_confidence = classification.confidence
            except Exception:
                logger.warning("Root cause analysis failed, using original classification")

        # Normalize and validate
        primary_method = _normalize_method(primary_method)
        if primary_method not in VALID_METHODS:
            logger.warning("Classifier returned unknown method '%s', defaulting", primary_method)
            primary_method = "technical_contradiction"

    # Step 5: Dispatch
    pipeline = _get_pipeline(primary_method)
    result = pipeline(
        problem_text, ideal_final_result, llm_client, store, research_tools=research_tools
    )

    # Step 6: Attach metadata
    result.method_confidence = method_confidence
    if secondary_method:
        normalized_secondary = _normalize_method(secondary_method)
        if normalized_secondary in VALID_METHODS:
            result.secondary_method = normalized_secondary

    return result
