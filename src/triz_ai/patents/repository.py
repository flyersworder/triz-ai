"""Pluggable patent repository protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from triz_ai.patents.store import (
        CandidateParameter,
        CandidatePrinciple,
        Classification,
        Patent,
    )


@runtime_checkable
class PatentRepository(Protocol):
    """Protocol for patent data storage backends.

    The default implementation is PatentStore (SQLite-backed).
    Alternative backends (Postgres, DynamoDB, etc.) implement this protocol
    to provide full database portability.
    """

    def init_db(self, force: bool = False) -> None: ...
    def close(self) -> None: ...

    # --- Patents ---
    def insert_patent(self, patent: Patent, embedding: list[float] | None = None) -> None: ...
    def get_patent(self, patent_id: str) -> Patent | None: ...
    def get_all_patents(self) -> list[Patent]: ...
    def search_patents(
        self, query_embedding: list[float], limit: int = 5
    ) -> list[tuple[Patent, float]]: ...
    def search_patents_hybrid(
        self,
        query_embedding: list[float],
        principle_ids: list[int] | None = None,
        improving_param: int | None = None,
        worsening_param: int | None = None,
        limit: int = 5,
    ) -> list[tuple[Patent, float]]: ...

    # --- Classifications ---
    def insert_classification(self, classification: Classification) -> None: ...
    def get_classification(self, patent_id: str) -> Classification | None: ...
    def get_unclassified_patents(self) -> list[Patent]: ...
    def get_classifications_by_domain(
        self, domain: str
    ) -> list[tuple[Patent, Classification]]: ...

    # --- Candidate Principles ---
    def insert_candidate_principle(self, candidate: CandidatePrinciple) -> None: ...
    def get_pending_candidates(self) -> list[CandidatePrinciple]: ...
    def update_candidate_status(self, candidate_id: str, status: str) -> None: ...

    # --- Candidate Parameters ---
    def insert_candidate_parameter(self, candidate: CandidateParameter) -> None: ...
    def get_pending_candidate_parameters(self) -> list[CandidateParameter]: ...
    def update_candidate_parameter_status(self, candidate_id: str, status: str) -> None: ...

    # --- Matrix Observations ---
    def insert_matrix_observation(
        self,
        improving: int,
        worsening: int,
        principle_id: int,
        patent_id: str,
        confidence: float,
    ) -> None: ...
    def get_matrix_observations(
        self, min_count: int = 3
    ) -> dict[tuple[int, int], list[tuple[int, int, float]]]: ...
