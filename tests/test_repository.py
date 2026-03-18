"""Tests for PatentRepository protocol conformance."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from triz_ai.patents.repository import PatentRepository
from triz_ai.patents.store import Patent, PatentStore

if TYPE_CHECKING:
    # Static conformance check — verified by type checker (uvx ty check),
    # catches signature mismatches that runtime_checkable cannot.
    def _assert_patent_store_conforms() -> None:
        _store: PatentRepository = PatentStore(db_path=":memory:")  # noqa: F841


@pytest.fixture
def store(tmp_path):
    """Create a temporary patent store."""
    db_path = tmp_path / "test.db"
    s = PatentStore(db_path=db_path)
    s.init_db()
    yield s
    s.close()


class TestProtocolConformance:
    """Verify PatentStore satisfies the PatentRepository protocol."""

    def test_isinstance_check(self, store):
        """PatentStore must pass runtime_checkable isinstance test."""
        assert isinstance(store, PatentRepository)

    def test_all_protocol_methods_exist(self, store):
        """PatentStore must have every method declared in PatentRepository."""
        protocol_methods = [
            "init_db",
            "close",
            "insert_patent",
            "get_patent",
            "get_all_patents",
            "search_patents",
            "search_patents_hybrid",
            "insert_classification",
            "get_classification",
            "get_unclassified_patents",
            "get_classifications_by_domain",
            "insert_candidate_principle",
            "get_pending_candidates",
            "update_candidate_status",
            "insert_candidate_parameter",
            "get_pending_candidate_parameters",
            "update_candidate_parameter_status",
            "insert_matrix_observation",
            "get_matrix_observations",
        ]
        for method_name in protocol_methods:
            assert hasattr(store, method_name), f"Missing method: {method_name}"
            assert callable(getattr(store, method_name)), f"Not callable: {method_name}"


class TestMockDelegation:
    """Verify consumers can work with any PatentRepository implementation."""

    def test_mock_repo_search_patents(self):
        """A mock satisfying PatentRepository can be used in place of PatentStore."""
        mock_repo = MagicMock(spec=PatentRepository)
        sample_patent = Patent(
            id="US999",
            title="Mock Patent",
            abstract="A mock patent",
        )
        mock_repo.search_patents.return_value = [(sample_patent, 0.95)]

        # Simulate what analyzer.search_patents does
        results = mock_repo.search_patents(query_embedding=[0.1] * 768, limit=5)

        mock_repo.search_patents.assert_called_once_with(query_embedding=[0.1] * 768, limit=5)
        assert len(results) == 1
        assert results[0][0].id == "US999"

    def test_mock_repo_insert_and_get(self):
        """Mock repo can handle insert + get patent round-trip."""
        mock_repo = MagicMock(spec=PatentRepository)
        patent = Patent(id="US001", title="Test", abstract="Abstract")
        mock_repo.get_patent.return_value = patent

        mock_repo.insert_patent(patent, embedding=[0.1] * 768)
        result = mock_repo.get_patent("US001")

        mock_repo.insert_patent.assert_called_once_with(patent, embedding=[0.1] * 768)
        assert result is not None
        assert result.id == "US001"

    def test_mock_repo_isinstance(self):
        """A mock with spec=PatentRepository passes isinstance check."""
        mock_repo = MagicMock(spec=PatentRepository)
        assert isinstance(mock_repo, PatentRepository)


class TestProtocolEnforcement:
    """Verify protocol rejects non-conforming implementations."""

    def test_incomplete_impl_fails_isinstance(self):
        """An object missing protocol methods must fail isinstance check."""

        class Incomplete:
            def init_db(self): ...

        assert not isinstance(Incomplete(), PatentRepository)
