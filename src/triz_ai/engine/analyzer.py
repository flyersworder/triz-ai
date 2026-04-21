"""TRIZ problem analysis pipeline."""

import logging

from pydantic import BaseModel

from triz_ai.knowledge.contradictions import lookup_with_observations
from triz_ai.knowledge.parameters import load_parameters
from triz_ai.knowledge.principles import load_principles
from triz_ai.llm.client import LLMClient
from triz_ai.patents.repository import PatentRepository

logger = logging.getLogger(__name__)


class AnalysisResult(BaseModel):
    """Result of TRIZ analysis of a problem.

    The result is tool-agnostic at the top level, with tool-specific data in `details`.
    For backward compatibility, contradiction-specific fields are kept but optional.
    `enrichment` holds data from enrichment-stage research tools (empty if none).
    """

    problem: str
    method: str = "technical_contradiction"
    method_confidence: float = 1.0
    secondary_method: str | None = None
    ideal_final_result: str | None = None

    # Contradiction-specific (kept for backward compat, populated from details)
    improving_param: dict | None = None  # {"id": int, "name": str}
    worsening_param: dict | None = None  # {"id": int, "name": str}
    reasoning: str = ""
    contradiction_confidence: float = 1.0
    recommended_principles: list[dict] = []  # [{"id": int, "name": str, "description": str}]

    # Common across all methods
    patent_examples: list[dict] = []
    solution_directions: list[dict] = []
    enrichment: list[dict] = []  # Data from enrichment-stage research tools
    details: dict = {}  # Tool-specific data


def analyze_contradiction(
    problem_text: str,
    ideal_final_result: str | None,
    llm_client: LLMClient,
    store: PatentRepository | None = None,
    research_tools: list | None = None,
) -> AnalysisResult:
    """Technical contradiction analysis pipeline.

    Pipeline:
    1. LLM extracts the technical contradiction
    2. Maps to engineering parameters
    3. Looks up contradiction matrix for recommended principles
    4. Searches patent store for examples (search-stage research tools run here)
    5. Generates solution directions
    6. Runs enrichment-stage research tools
    """
    # Step 1: Extract contradiction
    contradiction = llm_client.extract_contradiction(problem_text)

    # Step 2: Map to parameters
    parameters = {p.id: p for p in load_parameters()}
    improving = parameters.get(contradiction.improving_param)
    worsening = parameters.get(contradiction.worsening_param)

    if not improving or not worsening:
        raise ValueError(
            f"Invalid parameters: improving={contradiction.improving_param}, "
            f"worsening={contradiction.worsening_param}"
        )

    # Step 3: Lookup matrix (merges static + patent observations when store available)
    principle_ids = lookup_with_observations(
        contradiction.improving_param, contradiction.worsening_param, store=store
    )

    # Map to principle details
    all_principles = {p.id: p for p in load_principles()}
    recommended_principles = []
    for pid in principle_ids:
        p = all_principles.get(pid)
        if p:
            recommended_principles.append(
                {"id": p.id, "name": p.name, "description": p.description}
            )

    # Step 4: Hybrid patent search (if store available)
    patent_examples = search_patents(
        problem_text,
        llm_client,
        store,
        principle_ids=[p["id"] for p in recommended_principles],
        improving_param=contradiction.improving_param,
        worsening_param=contradiction.worsening_param,
        research_tools=research_tools,
    )

    # Step 5: Generate solution directions
    solution_directions = []
    if recommended_principles:
        try:
            directions = llm_client.generate_solution_directions(
                problem_text,
                improving_param=improving.name,
                worsening_param=worsening.name,
                principles=recommended_principles,
                patent_examples=patent_examples,
            )
            solution_directions = [d.model_dump() for d in directions.directions]
        except Exception:
            logger.warning(
                "Solution direction generation failed, continuing without", exc_info=True
            )

    improving_dict = {"id": improving.id, "name": improving.name}
    worsening_dict = {"id": worsening.id, "name": worsening.name}

    # Run enrichment tools
    enrichment = run_enrichment_tools(problem_text, solution_directions, research_tools)

    return AnalysisResult(
        problem=problem_text,
        method="technical_contradiction",
        ideal_final_result=ideal_final_result,
        improving_param=improving_dict,
        worsening_param=worsening_dict,
        reasoning=contradiction.reasoning,
        contradiction_confidence=contradiction.confidence,
        recommended_principles=recommended_principles,
        patent_examples=patent_examples,
        solution_directions=solution_directions,
        enrichment=enrichment,
        details={
            "improving_param": improving_dict,
            "worsening_param": worsening_dict,
            "reasoning": contradiction.reasoning,
            "contradiction_confidence": contradiction.confidence,
            "recommended_principles": recommended_principles,
        },
    )


def search_patents(
    problem_text: str,
    llm_client: LLMClient,
    store: PatentRepository | None,
    principle_ids: list[int] | None = None,
    improving_param: int | None = None,
    worsening_param: int | None = None,
    research_tools: list | None = None,
) -> list[dict]:
    """Search patent store and search-stage research tools for relevant examples.

    Research tools are filtered to those with "search" in their stages.
    Each tool receives a context dict with principle_ids, improving_param,
    and worsening_param when available. Results are deduplicated by title.
    """
    patent_examples: list[dict] = []

    # 1. Local DB search (existing logic)
    if store is not None:
        try:
            query_embedding = llm_client.get_embedding(problem_text)
            if principle_ids and improving_param and worsening_param:
                results = store.search_patents_hybrid(
                    query_embedding,
                    principle_ids=principle_ids,
                    improving_param=improving_param,
                    worsening_param=worsening_param,
                    limit=5,
                )
            else:
                results = store.search_patents(query_embedding, limit=5)

            all_principles_map = {p.id: p for p in load_principles()}
            for patent, _score in results:
                matched_principles = []
                if principle_ids:
                    classification = store.get_classification(patent.id)
                    if classification:
                        overlap = set(principle_ids) & set(classification.principle_ids)
                        matched_principles = [
                            all_principles_map[pid].name
                            for pid in overlap
                            if pid in all_principles_map
                        ]
                patent_examples.append(
                    {
                        "id": patent.id,
                        "title": patent.title,
                        "abstract": patent.abstract or "",
                        "assignee": patent.assignee,
                        "filing_date": patent.filing_date,
                        "matched_principles": matched_principles,
                    }
                )
        except Exception:
            logger.warning("Patent search failed, continuing without examples", exc_info=True)

    # 2. Research tools (search stage)
    if research_tools:
        seen_titles = {p["title"].lower() for p in patent_examples}
        search_context: dict = {"stage": "search"}
        if principle_ids:
            search_context["principle_ids"] = principle_ids
        if improving_param:
            search_context["improving_param"] = improving_param
        if worsening_param:
            search_context["worsening_param"] = worsening_param

        for tool in research_tools:
            if "search" not in tool.stages:
                continue
            try:
                tool_results = tool.fn(problem_text, search_context)
                for item in tool_results:
                    title = item.get("title", "")
                    if not title or title.lower() in seen_titles:
                        continue
                    seen_titles.add(title.lower())
                    patent_examples.append(
                        {
                            "id": item.get("id", ""),
                            "title": title,
                            "abstract": item.get("abstract", ""),
                            "assignee": item.get("assignee"),
                            "filing_date": item.get("filing_date"),
                            "url": item.get("url"),
                            "matched_principles": item.get("matched_principles", []),
                            "source": tool.name,
                        }
                    )
            except Exception:
                logger.warning("Research tool '%s' failed, skipping", tool.name, exc_info=True)

    return patent_examples


def run_enrichment_tools(
    problem_text: str,
    solution_directions: list[dict],
    research_tools: list | None = None,
) -> list[dict]:
    """Run enrichment-stage research tools after solution generation."""
    if not research_tools:
        return []
    from triz_ai.tools import run_stage_tools

    return run_stage_tools(
        research_tools,
        "enrichment",
        problem_text,
        extra_context={"solution_directions": solution_directions},
    )


def analyze(
    problem_text: str,
    llm_client: LLMClient | None = None,
    store: PatentRepository | None = None,
) -> AnalysisResult:
    """Analyze a technical problem using TRIZ methodology.

    Legacy entry point — delegates to analyze_contradiction for backward compatibility.
    """
    if llm_client is None:
        llm_client = LLMClient()

    return analyze_contradiction(problem_text, None, llm_client, store)
