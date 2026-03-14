"""LLM client wrapping litellm for TRIZ operations."""

import json
import logging
from typing import TypeVar

import litellm
from pydantic import BaseModel

from triz_ai.config import load_config
from triz_ai.llm.prompts import (
    classify_patent_prompt,
    cluster_patents_prompt,
    extract_contradiction_prompt,
    generate_ideas_prompt,
    propose_candidate_principle_prompt,
)

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
        self.api_base = config.llm.api_base
        self.api_key = config.llm.api_key
        self.embedding_model = config.embeddings.model
        self.embedding_dimensions = config.embeddings.dimensions
        self.embedding_api_base = config.embeddings.api_base
        self.embedding_api_key = config.embeddings.api_key

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
            kwargs: dict = {}
            if self.api_base:
                kwargs["api_base"] = self.api_base
            if self.api_key:
                kwargs["api_key"] = self.api_key
            response = litellm.completion(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                **kwargs,
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
        return self._complete(
            extract_contradiction_prompt(),
            problem_text,
            ExtractedContradiction,
        )

    def classify_patent(self, patent_text: str) -> PatentClassification:
        """Classify a patent by TRIZ principles."""
        return self._complete(
            classify_patent_prompt(),
            patent_text,
            PatentClassification,
        )

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

    def propose_candidate_principle(self, patent_cluster: list[str]) -> CandidatePrincipleProposal:
        """Propose a candidate new TRIZ principle from a cluster of patents."""
        patents_text = "\n---\n".join(patent_cluster)
        return self._complete(
            propose_candidate_principle_prompt(),
            patents_text,
            CandidatePrincipleProposal,
        )

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text."""
        kwargs: dict = {}
        if self.embedding_api_base:
            kwargs["api_base"] = self.embedding_api_base
        if self.embedding_api_key:
            kwargs["api_key"] = self.embedding_api_key
        response = litellm.embedding(
            model=self.embedding_model,
            input=[text],
            dimensions=self.embedding_dimensions,
            **kwargs,
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

        result = self._complete(cluster_patents_prompt(), abstracts_text, ClusterResult)
        return result.clusters
