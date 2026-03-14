"""Prompt templates with TRIZ context injection.

Each builder function injects only the relevant TRIZ knowledge to keep
system prompts under ~2K tokens.
"""

from triz_ai.knowledge.parameters import load_parameters
from triz_ai.knowledge.principles import load_principles


def _parameters_list() -> str:
    """Compact list of all 39 parameters: 'ID. Name' per line."""
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
        '{"improving_param": <int 1-39>, "worsening_param": <int 1-39>, '
        '"reasoning": "<brief explanation of the contradiction>"}'
    )


def classify_patent_prompt() -> str:
    """System prompt for classifying patents by TRIZ principles."""
    principles = _principles_compact()
    return (
        "You are a TRIZ expert analyzing patents.\n\n"
        "Identify which TRIZ inventive principles this patent employs, "
        "what technical contradiction it resolves (using engineering parameter "
        "IDs 1-39), and your confidence.\n\n"
        "TRIZ Inventive Principles:\n"
        f"{principles}\n\n"
        "Respond with JSON:\n"
        '{"principle_ids": [<int>], '
        '"contradiction": {"improving": <int 1-39>, "worsening": <int 1-39>}, '
        '"confidence": <float 0.0-1.0>, '
        '"reasoning": "<brief explanation>"}'
    )


def generate_ideas_prompt() -> str:
    """System prompt for idea generation."""
    return (
        "You are a TRIZ innovation expert generating novel ideas by applying "
        "underused TRIZ principles to a specific technology domain.\n\n"
        "For each idea:\n"
        "- Apply a specific underused principle in a concrete, actionable way\n"
        "- Consider how existing patents in the domain might be improved or extended\n"
        "- Be specific and technical, not generic\n\n"
        "Respond with JSON:\n"
        '{"ideas": [{"idea": "<concrete technical description>", '
        '"principle_id": <int>, '
        '"reasoning": "<why this principle creates novelty here>"}]}'
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
