"""Prompt templates with TRIZ context injection.

Each builder function injects only the relevant TRIZ knowledge to keep
system prompts under ~2K tokens.
"""

from triz_ai.knowledge.parameters import load_parameters
from triz_ai.knowledge.principles import load_principles
from triz_ai.knowledge.separation import load_separation_principles
from triz_ai.knowledge.solutions import load_standard_solutions
from triz_ai.knowledge.trends import load_evolution_trends


def _parameters_list() -> str:
    """Compact list of all 50 parameters: 'ID. Name' per line."""
    return "\n".join(f"{p.id}. {p.name}" for p in load_parameters())


def _principles_compact() -> str:
    """Compact list of all 40 principles: 'ID. Name — Description' per line."""
    return "\n".join(f"{p.id}. {p.name} — {p.description}" for p in load_principles())


def _principles_subset(principle_ids: list[int]) -> str:
    """Detailed view of specific principles including sub-principles and keywords."""
    principles = {p.id: p for p in load_principles()}
    parts = []
    for pid in principle_ids:
        p = principles.get(pid)
        if p:
            subs = "; ".join(p.sub_principles)
            keywords = ", ".join(p.keywords)
            parts.append(
                f"Principle {p.id}: {p.name}\n"
                f"  Description: {p.description}\n"
                f"  Sub-principles: {subs}\n"
                f"  Keywords: {keywords}"
            )
    return "\n\n".join(parts)


def extract_contradiction_prompt() -> str:
    """System prompt for extracting technical contradictions."""
    params = _parameters_list()
    return (
        "You are a TRIZ (Theory of Inventive Problem Solving) expert.\n\n"
        "Analyze the technical problem and identify the core technical contradiction: "
        "which engineering parameter is the user trying to IMPROVE, and which parameter "
        "WORSENS as a result?\n\n"
        "Map both to the closest parameters from this list:\n\n"
        f"{params}\n\n"
        "Respond with JSON:\n"
        '{"improving_param": <int 1-50>, "worsening_param": <int 1-50>, '
        '"reasoning": "<brief explanation of the contradiction>", '
        '"confidence": <float 0.0-1.0, how confident you are in this mapping>}'
    )


def classify_patent_prompt() -> str:
    """System prompt for classifying patents by TRIZ principles."""
    principles = _principles_compact()
    return (
        "You are a TRIZ expert analyzing patents.\n\n"
        "Identify which TRIZ inventive principles this patent employs, "
        "what technical contradiction it resolves (using engineering parameter "
        "IDs 1-50), and your confidence.\n\n"
        "TRIZ Inventive Principles:\n"
        f"{principles}\n\n"
        "Respond with JSON:\n"
        '{"principle_ids": [<int>], '
        '"contradiction": {"improving": <int 1-50>, "worsening": <int 1-50>}, '
        '"confidence": <float 0.0-1.0>, '
        '"reasoning": "<brief explanation>"}'
    )


def solution_directions_prompt() -> str:
    """System prompt for generating concrete solution directions from TRIZ analysis."""
    return (
        "You are a TRIZ (Theory of Inventive Problem Solving) expert helping engineers "
        "develop concrete solution directions.\n\n"
        "Given a technical problem, its contradiction, recommended TRIZ principles, and "
        "related patents, generate 2-3 concrete solution directions that:\n"
        "- Apply the recommended principles to the specific problem\n"
        "- Are actionable and specific (not generic advice)\n"
        "- Reference how similar problems were solved in the patents when relevant\n\n"
        "Respond with JSON:\n"
        '{"directions": [{"title": "<short title>", '
        '"description": "<2-3 sentences explaining the approach>", '
        '"principles_applied": ["<principle name>", ...]}]}'
    )


def generate_ideas_prompt() -> str:
    """System prompt for idea generation."""
    return (
        "You are a TRIZ innovation expert generating novel ideas by applying "
        "underused TRIZ principles to a specific technology domain.\n\n"
        "For each idea:\n"
        "- Apply a specific underused principle in a concrete, actionable way\n"
        "- Consider how existing patents in the domain might be improved or extended\n"
        "- Be specific and technical, not generic\n"
        "- If inspired by a specific patent, include its ID as source_patent_id\n\n"
        "Respond with JSON:\n"
        '{"ideas": [{"idea": "<concrete technical description>", '
        '"principle_id": <int>, '
        '"reasoning": "<why this principle creates novelty here>", '
        '"source_patent_id": "<patent ID or null>"}]}'
    )


def propose_candidate_principle_prompt() -> str:
    """System prompt for proposing candidate new principles."""
    principles = _principles_compact()
    return (
        "You are a TRIZ methodology researcher. The following patents share a common "
        "inventive pattern that does NOT map well to any of the existing 40 TRIZ "
        "principles listed below.\n\n"
        "Existing TRIZ Principles (do NOT duplicate these):\n"
        f"{principles}\n\n"
        "Analyze the shared inventive pattern in the patents and propose a candidate "
        "NEW principle that captures this pattern.\n\n"
        "Respond with JSON:\n"
        '{"name": "<concise principle name>", '
        '"description": "<what the principle is and when to apply it>", '
        '"how_it_differs": "<specifically how it differs from the closest existing principles>", '
        '"confidence": <float 0.0-1.0>}'
    )


def propose_candidate_parameter_prompt() -> str:
    """System prompt for proposing candidate new engineering parameters."""
    params = _parameters_list()
    return (
        "You are a TRIZ methodology researcher. The following patents involve "
        "technical contradictions that do NOT map well to any of the existing "
        "engineering parameters listed below.\n\n"
        "Existing TRIZ Engineering Parameters (do NOT duplicate these):\n"
        f"{params}\n\n"
        "Analyze the shared contradiction pattern in the patents and propose a "
        "candidate NEW engineering parameter that captures the dimension of "
        "improvement or degradation these patents have in common.\n\n"
        "Respond with JSON:\n"
        '{"name": "<concise parameter name>", '
        '"description": "<what the parameter measures and when it applies>", '
        '"how_it_differs": "<specifically how it differs from the closest existing parameters>", '
        '"confidence": <float 0.0-1.0>}'
    )


def cluster_patents_prompt() -> str:
    """System prompt for LLM-based patent clustering."""
    return (
        "You are analyzing patents to find common inventive patterns that don't fit "
        "well into the standard 40 TRIZ principles.\n\n"
        "Group the patent abstracts below into clusters based on shared inventive "
        "patterns. Requirements:\n"
        "- Each cluster must have at least 3 patents\n"
        "- Patents that don't fit any cluster should be excluded\n"
        "- Focus on the inventive MECHANISM, not the domain\n\n"
        "Respond with JSON:\n"
        '{"clusters": [[0, 1, 2], [3, 4, 5]], '
        '"cluster_descriptions": ["description of shared pattern 1", '
        '"description of shared pattern 2"]}'
    )


def seed_matrix_prompt(
    improving_param_id: int,
    improving_param_name: str,
    worsening_params: list[dict],
    example_rows: list[str],
) -> str:
    """System prompt for LLM-seeding missing contradiction matrix cells."""
    params = _parameters_list()
    principles = _principles_compact()
    examples = "\n".join(example_rows)
    worsening_list = "\n".join(f"- {wp['id']}. {wp['name']}" for wp in worsening_params)

    return (
        "You are a TRIZ (Theory of Inventive Problem Solving) expert.\n\n"
        "Your task is to suggest the most applicable TRIZ inventive principles "
        "for each (improving, worsening) parameter pair in the contradiction matrix.\n\n"
        "TRIZ Engineering Parameters (1-50):\n"
        f"{params}\n\n"
        "TRIZ Inventive Principles (1-40):\n"
        f"{principles}\n\n"
        "Here are example rows from the existing matrix to demonstrate the format:\n"
        f"{examples}\n\n"
        f"Now fill in the matrix cells for improving parameter "
        f"{improving_param_id} ({improving_param_name}) against each of these "
        f"worsening parameters:\n{worsening_list}\n\n"
        "Rules:\n"
        "- Suggest up to 4 principle IDs (1-40) per cell\n"
        "- Choose principles that best resolve the contradiction between improving "
        "and worsening parameters\n"
        "- If no principles apply well, use an empty list\n\n"
        "Respond with JSON:\n"
        '{"entries": [{"improving": <int>, "worsening": <int>, '
        '"principles": [<int>, ...]}]}'
    )


# --- Multi-tool routing prompts ---


def _separation_principles_text() -> str:
    """Compact list of separation principles with techniques."""
    parts = []
    for sp in load_separation_principles():
        techniques = "; ".join(sp.techniques)
        parts.append(
            f"{sp.id}. {sp.name} ({sp.category}): {sp.description}\n   Techniques: {techniques}"
        )
    return "\n".join(parts)


def _standard_solutions_compact() -> str:
    """Compact list of standard solutions grouped by class."""
    solutions = load_standard_solutions()
    parts = []
    current_class = None
    for s in solutions:
        if s.class_id != current_class:
            current_class = s.class_id
            parts.append(f"\nClass {s.class_id}: {s.class_name}")
        parts.append(f"  {s.id}. {s.name} — {s.description}")
    return "\n".join(parts)


def _evolution_trends_text() -> str:
    """Compact list of evolution trends with stages."""
    parts = []
    for t in load_evolution_trends():
        stages = "; ".join(f"Stage {s.stage}: {s.name}" for s in t.stages)
        parts.append(f"{t.id}. {t.name}: {t.description}\n   Stages: {stages}")
    return "\n".join(parts)


def classify_problem_prompt() -> str:
    """System prompt for classifying a problem into the appropriate TRIZ tool."""
    return (
        "You are a TRIZ (Theory of Inventive Problem Solving) expert.\n\n"
        "Classify the engineering problem into the most appropriate TRIZ analysis method:\n\n"
        "1. **technical_contradiction** — The problem is about improving one parameter that "
        "worsens another. Examples: 'increase speed without increasing weight', "
        "'improve strength without reducing flexibility', 'increase throughput without "
        "increasing error rate'.\n\n"
        "2. **physical_contradiction** — A single component must have two opposite properties "
        "simultaneously. Examples: 'the solder joint must be rigid AND flexible', "
        "'the surface must be smooth AND rough', 'the liquid must be hot AND cold'.\n\n"
        "3. **su_field** — The problem involves detection, measurement, or insufficient/harmful "
        "interactions between substances and fields. Examples: 'how to detect cracks without "
        "adding sensors', 'how to measure internal temperature non-invasively', "
        "'the field damages the substrate'.\n\n"
        "4. **function_analysis** — A component performs a harmful function, or the user wants "
        "to understand which interactions are problematic. Examples: 'the adhesive damages the "
        "die', 'which component is causing failures', 'the cooling system corrodes the pipes'.\n\n"
        "5. **trimming** — The goal is simplification, cost reduction, or removing components. "
        "Examples: 'reduce BOM cost', 'simplify the assembly', 'eliminate unnecessary parts', "
        "'reduce component count'.\n\n"
        "6. **trends** — The question is about future technology evolution or generational "
        "progression. Examples: 'what is the next generation of X', 'where is this technology "
        "heading', 'how will X evolve'.\n\n"
        "Also suggest a secondary method that might offer additional insights.\n\n"
        "Reformulate the problem statement to be more precise and actionable.\n\n"
        "Respond with JSON:\n"
        '{"primary_method": "<method>", "secondary_method": "<method or null>", '
        '"reasoning": "<why this method fits best>", '
        '"confidence": <float 0.0-1.0>, '
        '"reformulated_problem": "<clearer problem statement>"}'
    )


def ideal_final_result_prompt() -> str:
    """System prompt for formulating the Ideal Final Result."""
    return (
        "You are a TRIZ expert. Formulate the Ideal Final Result (IFR) for this problem.\n\n"
        "The IFR describes the perfect solution where:\n"
        "- The useful function is fully achieved\n"
        "- No new harmful effects are introduced\n"
        "- No additional complexity, cost, or resources are needed\n"
        "- The system achieves the goal 'by itself'\n\n"
        "Express the IFR as: 'The [system element] ITSELF [performs the desired action] "
        "without [any negative consequences], using only [available resources].'\n\n"
        "Respond with JSON:\n"
        '{"ideal_result": "<the IFR statement>", '
        '"reasoning": "<why this is the ideal outcome>"}'
    )


def root_cause_analysis_prompt() -> str:
    """System prompt for root cause analysis when problem is vague."""
    return (
        "You are a TRIZ expert. The problem description is vague or ambiguous.\n\n"
        "Apply root cause analysis to trace the problem to its fundamental cause:\n"
        "1. Ask 'Why?' repeatedly (5-Whys approach)\n"
        "2. Identify the chain of causes\n"
        "3. Find the root technical contradiction or physical limitation\n"
        "4. Reformulate the problem as a clear, specific engineering challenge\n\n"
        "Respond with JSON:\n"
        '{"root_causes": ["<cause 1 (surface)>", "<cause 2 (deeper)>", "..."], '
        '"reformulated_problem": "<specific engineering problem statement>", '
        '"reasoning": "<how you traced to the root cause>"}'
    )


def extract_physical_contradiction_prompt() -> str:
    """System prompt for extracting physical contradictions."""
    sep_principles = _separation_principles_text()
    return (
        "You are a TRIZ expert analyzing a physical contradiction.\n\n"
        "A physical contradiction exists when a single element must have two opposite "
        "properties simultaneously (e.g., hot AND cold, rigid AND flexible).\n\n"
        "Identify:\n"
        "1. The property that has contradictory requirements\n"
        "2. The two opposing requirements (requirement_a and requirement_b)\n"
        "3. The best separation principle to resolve it\n\n"
        "Separation Principles:\n"
        f"{sep_principles}\n\n"
        "Choose the most applicable separation type and list specific techniques.\n\n"
        "Respond with JSON:\n"
        '{"property": "<the conflicting property>", '
        '"requirement_a": "<first requirement>", '
        '"requirement_b": "<opposite requirement>", '
        '"separation_type": "<separation_in_time|separation_in_space|'
        'separation_in_scale|separation_upon_condition>", '
        '"separation_principles": [{"id": <int>, "name": "<name>", '
        '"technique": "<specific technique from the list>"}]}'
    )


def su_field_analysis_prompt() -> str:
    """System prompt for Su-Field analysis."""
    solutions = _standard_solutions_compact()
    return (
        "You are a TRIZ expert performing Su-Field (Substance-Field) analysis.\n\n"
        "A Su-Field model consists of:\n"
        "- S1 (substance 1): the object being acted upon\n"
        "- S2 (substance 2): the tool acting on S1\n"
        "- F (field): the energy/interaction connecting S1 and S2\n\n"
        "Problem types:\n"
        "- incomplete: missing substance or field in the model\n"
        "- harmful: the interaction produces undesirable effects\n"
        "- inefficient: the interaction exists but is too weak\n\n"
        "Standard Solutions:\n"
        f"{solutions}\n\n"
        "Identify the Su-Field model elements, classify the problem type, "
        "and recommend the most applicable standard solutions.\n\n"
        "Respond with JSON:\n"
        '{"substances": ["<S1>", "<S2>"], "field": "<field type>", '
        '"problem_type": "<incomplete|harmful|inefficient>", '
        '"standard_solutions": [{"id": "<e.g. 1.1.1>", "name": "<name>", '
        '"applicability": "<why this solution applies>"}]}'
    )


def function_analysis_prompt() -> str:
    """System prompt for function analysis."""
    return (
        "You are a TRIZ expert performing function analysis.\n\n"
        "Decompose the system into components and their functions. For each function:\n"
        "- Subject: the component performing the action\n"
        "- Action: what it does (verb)\n"
        "- Object: what it acts upon\n"
        "- Type: useful | harmful | insufficient | excessive\n\n"
        "Identify problematic functions (harmful, insufficient, or excessive) "
        "and recommend how to resolve them using TRIZ approaches:\n"
        "- Harmful → eliminate, shield, or counteract\n"
        "- Insufficient → enhance, add resources, or restructure\n"
        "- Excessive → reduce, limit, or redistribute\n\n"
        "Respond with JSON:\n"
        '{"components": [{"name": "<name>", "role": "<brief role>"}], '
        '"functions": [{"subject": "<component>", "action": "<verb>", '
        '"object": "<component>", "type": "<useful|harmful|insufficient|excessive>"}], '
        '"problem_functions": [{"subject": "<component>", "action": "<verb>", '
        '"object": "<component>", "problem": "<what is wrong>"}], '
        '"recommendations": ["<recommendation 1>", "..."]}'
    )


def trimming_analysis_prompt() -> str:
    """System prompt for trimming analysis."""
    return (
        "You are a TRIZ expert performing trimming analysis.\n\n"
        "Trimming simplifies a system by removing components and redistributing "
        "their useful functions to remaining components or the supersystem.\n\n"
        "Trimming rules:\n"
        "A. The function is not needed → remove the component entirely\n"
        "B. The function can be performed by another existing component → remove and reassign\n"
        "C. The function can be performed by the object itself → remove (self-service)\n\n"
        "For each component, assess:\n"
        "- What useful function does it perform?\n"
        "- How expensive/complex is it? (high/medium/low)\n"
        "- Can its function be eliminated or performed by something else?\n\n"
        "Respond with JSON:\n"
        '{"components": [{"name": "<name>", "function": "<primary function>", '
        '"cost": "<high|medium|low>"}], '
        '"trimming_candidates": [{"component": "<name>", '
        '"reason": "<why it can be trimmed>", '
        '"rule": "<A|B|C — which trimming rule applies>"}], '
        '"redistributed_functions": [{"function": "<function description>", '
        '"from": "<trimmed component>", "to": "<receiving component or self>"}]}'
    )


def trends_analysis_prompt() -> str:
    """System prompt for technology evolution trends analysis."""
    trends = _evolution_trends_text()
    return (
        "You are a TRIZ expert analyzing technology evolution using TRIZ trends "
        "and the System Operator (9-screen analysis).\n\n"
        "TRIZ Evolution Trends:\n"
        f"{trends}\n\n"
        "System Operator framework — analyze at three levels:\n"
        "- Subsystem: key internal components and their evolution\n"
        "- System: the technology as a whole\n"
        "- Supersystem: the broader environment and interacting systems\n\n"
        "For each level, consider past → present → future.\n\n"
        "Identify which evolution trend best describes the technology's current position, "
        "determine the current stage, and predict the next evolutionary steps.\n\n"
        "Respond with JSON:\n"
        '{"current_stage": {"trend_id": <int>, "trend_name": "<name>", '
        '"stage": <int>, "stage_name": "<name>"}, '
        '"trend_name": "<primary trend name>", '
        '"next_stages": [{"stage": <int>, "name": "<name>", '
        '"description": "<what this stage looks like for this technology>"}], '
        '"predictions": ["<prediction 1>", "<prediction 2>", "..."]}'
    )


# --- Deep ARIZ-85C prompts ---


def deep_reformulation_prompt() -> str:
    """System prompt implementing ARIZ Parts 1-3 (problem reformulation,
    contradiction intensification, and IFR formulation) in one LLM pass."""
    params = _parameters_list()
    return (
        "You are a TRIZ expert performing deep ARIZ-85C analysis (Parts 1-3).\n\n"
        "Given a problem description, perform the following steps:\n\n"
        "1. **Reformulate the problem** to reveal hidden contradictions. Strip away "
        "domain jargon and restate the problem in terms of conflicting requirements.\n\n"
        "2. **Identify TWO technical contradictions** (TC1 and TC2). TC1 is the primary "
        "contradiction; TC2 is the reversed/alternate formulation. Intensify each to "
        "its extreme — push the improving parameter to its maximum and observe the "
        "worst-case worsening.\n\n"
        "3. **Map each TC's parameters** to the engineering parameter list below "
        "(IDs 1-50).\n\n"
        "4. **State the physical contradiction** at both macro level (the part as a "
        "whole must be A AND not-A) and micro level (the particles/molecules of the "
        "part must be A AND not-A). If no physical contradiction exists, return null.\n\n"
        "5. **Formulate the Ideal Final Result (IFR)** in TRIZ format: "
        "'The [element] ITSELF [action] without [harm]'.\n\n"
        "6. **Inventory available resources**: substances present in or near the system, "
        "fields already acting, time resources (before/during/after), and space resources "
        "(inside/outside/surface).\n\n"
        "7. **Recommend 2-4 TRIZ tools** from: technical_contradiction, "
        "physical_contradiction, su_field, function_analysis, trimming, trends. "
        "Choose the tools most likely to resolve the identified contradictions.\n\n"
        "Engineering Parameters (1-50):\n"
        f"{params}\n\n"
        "Respond with JSON:\n"
        '{"original_problem": "<original>", '
        '"reformulated_problem": "<deeper reformulation>", '
        '"technical_contradiction_1": {'
        '"improving_param_id": <int>, "improving_param_name": "<name>", '
        '"worsening_param_id": <int>, "worsening_param_name": "<name>", '
        '"intensified_description": "<TC pushed to extreme>"}, '
        '"technical_contradiction_2": {'
        '"improving_param_id": <int>, "improving_param_name": "<name>", '
        '"worsening_param_id": <int>, "worsening_param_name": "<name>", '
        '"intensified_description": "<alternate TC>"}, '
        '"physical_contradiction": {'
        '"property": "<prop>", '
        '"macro_requirement": "<A>", '
        '"micro_requirement": "<B>"} or null, '
        '"ideal_final_result": "<IFR statement>", '
        '"resource_inventory": {'
        '"substances": ["<substance>", "..."], '
        '"fields": ["<field>", "..."], '
        '"time_resources": ["<time resource>", "..."], '
        '"space_resources": ["<space resource>", "..."]}, '
        '"recommended_tools": ["<method1>", "<method2>"], '
        '"reasoning": "<reasoning behind reformulation>"}'
    )


def solution_verification_prompt() -> str:
    """System prompt implementing ARIZ Part 7 (solution verification and
    synthesis across multiple candidate solutions)."""
    return (
        "You are a TRIZ expert performing ARIZ-85C solution verification (Part 7).\n\n"
        "Given the original problem, its Ideal Final Result (IFR), and a set of "
        "candidate solutions from different TRIZ tools, perform the following:\n\n"
        "1. **Verify each candidate** against the IFR:\n"
        "   - Does it fully satisfy the IFR?\n"
        "   - What gap remains between the candidate and the IFR?\n"
        "   - Score ideality (0.0-1.0) based on:\n"
        "     * Useful function achieved (0.0-0.4)\n"
        "     * No harmful side effects (0.0-0.3)\n"
        "     * Minimal resources consumed (0.0-0.3)\n\n"
        "2. **Synthesize combined solutions** by taking the best elements from "
        "multiple candidates. A synthesized solution may combine principles from "
        "different methods to get closer to the IFR.\n\n"
        "3. **Identify supersystem changes** needed — changes to the environment, "
        "interfaces, or adjacent systems required for each synthesized solution.\n\n"
        "Respond with JSON:\n"
        '{"verified_candidates": [{'
        '"method": "<method>", '
        '"satisfies_ifr": <bool>, '
        '"ifr_gap": "<what is missing>", '
        '"ideality_score": <float>, '
        '"key_insight": "<key insight>"}], '
        '"any_satisfies_ifr": <bool>, '
        '"synthesized_solutions": [{'
        '"title": "<title>", '
        '"description": "<description>", '
        '"principles_applied": ["<principle>", "..."], '
        '"supersystem_changes": ["<change>", "..."], '
        '"ideality_score": <float>}], '
        '"reasoning": "<verification reasoning>"}'
    )
