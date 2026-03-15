"""Prompt templates with TRIZ context injection.

Each builder function injects only the relevant TRIZ knowledge to keep
system prompts under ~2K tokens.
"""

from triz_ai.knowledge.parameters import load_parameters
from triz_ai.knowledge.principles import load_principles


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
