"""LLM client wrapping litellm for TRIZ operations."""

import json
import logging
from typing import TypeVar

import litellm
from pydantic import BaseModel

from triz_ai.config import load_config

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


# Response models
class ExtractedContradiction(BaseModel):
    improving_param: int
    worsening_param: int
    reasoning: str


class PatentClassification(BaseModel):
    principle_ids: list[int]
    contradiction: dict  # {"improving": int, "worsening": int}
    confidence: float
    reasoning: str


class Idea(BaseModel):
    idea: str
    principle_id: int
    reasoning: str


class IdeaBatch(BaseModel):
    ideas: list[Idea]


class CandidatePrincipleProposal(BaseModel):
    name: str
    description: str
    how_it_differs: str
    confidence: float


class LLMClient:
    def __init__(self, model: str | None = None):
        config = load_config()
        self.model = model or config.llm.default_model
        self.embedding_model = config.embeddings.model

    def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        retry: bool = True,
    ) -> T:
        """Call LLM and validate response against pydantic model.

        On malformed response, retry once with stricter prompt, then fail with raw response logged.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)
            return response_model.model_validate(data)
        except Exception as e:
            if retry:
                logger.warning(f"First LLM attempt failed ({e}), retrying with stricter prompt")
                strict_system = (
                    system_prompt
                    + f"\n\nIMPORTANT: You MUST respond with valid JSON matching this exact"
                    f" schema: {response_model.model_json_schema()}"
                )
                return self._complete(strict_system, user_prompt, response_model, retry=False)
            logger.error(f"LLM response validation failed: {e}")
            raise

    def extract_contradiction(self, problem_text: str) -> ExtractedContradiction:
        """Extract technical contradiction from problem description."""
        system_prompt = (
            "You are a TRIZ (Theory of Inventive Problem Solving) expert.\n"
            "Analyze the technical problem and identify the technical contradiction.\n"
            "Map it to the closest TRIZ engineering parameters (1-39).\n\n"
            'Respond with JSON: {"improving_param": <int 1-39>, "worsening_param": <int 1-39>,'
            ' "reasoning": "<explanation>"}'
        )
        return self._complete(system_prompt, problem_text, ExtractedContradiction)

    def classify_patent(self, patent_text: str) -> PatentClassification:
        """Classify a patent by TRIZ principles."""
        system_prompt = (
            "You are a TRIZ expert analyzing patents.\n"
            "Identify which TRIZ inventive principles (1-40) this patent employs,\n"
            "what contradiction it resolves, and your confidence level.\n\n"
            'Respond with JSON: {"principle_ids": [<ints>], "contradiction":'
            ' {"improving": <int>, "worsening": <int>}, "confidence": <float 0-1>,'
            ' "reasoning": "<explanation>"}'
        )
        return self._complete(system_prompt, patent_text, PatentClassification)

    def generate_ideas(
        self,
        domain: str,
        underused_principles: list[dict],
        existing_patents: list[str],
    ) -> IdeaBatch:
        """Generate novel ideas using underused principles."""
        principles_text = "\n".join(
            f"- Principle {p['id']}: {p['name']} - {p['description']}"
            for p in underused_principles
        )
        patents_text = "\n".join(f"- {p}" for p in existing_patents[:10])

        system_prompt = (
            "You are a TRIZ innovation expert. Generate novel ideas by applying"
            " underused TRIZ principles to a domain.\n\n"
            'Respond with JSON: {"ideas": [{"idea": "<description>",'
            ' "principle_id": <int>, "reasoning": "<why this principle applies>"}]}'
        )

        user_prompt = (
            f"Domain: {domain}\n\n"
            f"Underused TRIZ principles in this domain:\n{principles_text}\n\n"
            f"Existing patents in this domain:\n{patents_text}\n\n"
            f"Generate 3-5 novel ideas applying these underused principles to {domain}."
        )
        return self._complete(system_prompt, user_prompt, IdeaBatch)

    def propose_candidate_principle(self, patent_cluster: list[str]) -> CandidatePrincipleProposal:
        """Propose a candidate new TRIZ principle from a cluster of patents."""
        patents_text = "\n---\n".join(patent_cluster)

        system_prompt = (
            "You are a TRIZ methodology researcher. These patents share a common"
            " inventive pattern that doesn't map well to any existing TRIZ principle (1-40).\n\n"
            "Propose a candidate new principle that captures this pattern.\n\n"
            'Respond with JSON: {"name": "<principle name>", "description":'
            ' "<what the principle is>", "how_it_differs": "<how it differs from'
            ' existing 40 principles>", "confidence": <float 0-1>}'
        )

        return self._complete(system_prompt, patents_text, CandidatePrincipleProposal)

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text."""
        response = litellm.embedding(
            model=self.embedding_model,
            input=[text],
        )
        return response.data[0]["embedding"]

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

        system_prompt = (
            "You are analyzing patents to find common inventive patterns.\n"
            "Group the following patent abstracts into clusters based on shared"
            " inventive patterns that don't map well to existing TRIZ principles.\n"
            "Each cluster must have at least 3 patents. Patents that don't fit"
            " any cluster should be excluded.\n\n"
            'Respond with JSON: {"clusters": [[0, 1, 2], [3, 4, 5]],'
            ' "cluster_descriptions": ["description of pattern 1",'
            ' "description of pattern 2"]}'
        )

        result = self._complete(system_prompt, abstracts_text, ClusterResult)
        return result.clusters
