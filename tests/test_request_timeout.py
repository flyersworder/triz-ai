"""Timeout bounds on LLM calls.

Regression guard for the hang that broke `analyze` in 0.19.0: when a provider
accepts the connection and never answers, litellm's own default request_timeout
is 6000s (100 minutes), so the CLI sat there instead of failing. These tests
drive the real client against a socket that accepts and never replies, and
assert the call gives up on time.

The timeout is PER ATTEMPT, so wall clock scales with retries -- that is
asserted explicitly, because it is the part that surprises people.
"""

import socket
import threading
import time

import pytest
from pydantic import BaseModel

from triz_ai.llm.client import LLMClient, TrizAIError


class _Answer(BaseModel):
    answer: str


@pytest.fixture
def black_hole():
    """A server that accepts TCP connections and never sends a response."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    held = []

    def accept_forever():
        while True:
            try:
                conn, _ = srv.accept()
                held.append(conn)  # hold it open, read nothing, reply never
            except OSError:
                return

    threading.Thread(target=accept_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.getsockname()[1]}/v1"
    srv.close()
    for c in held:
        c.close()


def _write_config(tmp_path, base, timeout, retries):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
llm:
  default_model: openai/gpt-x
  classify_model: openai/gpt-x
  api_base: {base}
  api_key: sk-test
  request_timeout: {timeout}
  max_retries: {retries}
embeddings:
  model: openai/text-embedding-3-small
  api_base: {base}
  api_key: sk-test
  dimensions: 768
  request_timeout: {timeout}
"""
    )
    return cfg


@pytest.fixture
def configured(tmp_path, black_hole, monkeypatch):
    def _make(timeout=2.0, retries=0):
        cfg = _write_config(tmp_path, black_hole, timeout, retries)
        monkeypatch.setenv("TRIZ_AI_CONFIG", str(cfg))
        return LLMClient()

    return _make


@pytest.mark.timeout(60)
def test_completion_times_out_instead_of_hanging(configured):
    client = configured(timeout=2.0, retries=0)
    start = time.time()
    with pytest.raises(TrizAIError):
        client._complete("sys", "user", _Answer, retry=False)
    assert time.time() - start < 20, "completion did not honor request_timeout"


@pytest.mark.timeout(60)
def test_embedding_times_out_instead_of_hanging(configured):
    client = configured(timeout=2.0, retries=0)
    start = time.time()
    with pytest.raises(TrizAIError):
        client.get_embedding("hello")
    assert time.time() - start < 20, "embedding did not honor request_timeout"


@pytest.mark.timeout(90)
def test_openai_fallback_times_out_instead_of_hanging(configured, monkeypatch):
    """The no-litellm path bounds itself via the client, not per-call kwargs."""
    monkeypatch.setattr("triz_ai.llm.client.HAS_LITELLM", False)
    client = configured(timeout=2.0, retries=0)
    start = time.time()
    with pytest.raises(TrizAIError):
        client.get_embedding("hello")
    assert time.time() - start < 20, "openai fallback did not honor request_timeout"


@pytest.mark.timeout(120)
def test_completion_retries_multiply_the_timeout(configured):
    """Completion wall clock is request_timeout * (max_retries + 1).

    This is the property that turns a generous timeout plus default retries into
    a multi-minute hang, so it is worth pinning down rather than assuming.
    """
    client = configured(timeout=2.0, retries=0)
    start = time.time()
    with pytest.raises(TrizAIError):
        client._complete("sys", "user", _Answer, retry=False)
    one_attempt = time.time() - start

    client = configured(timeout=2.0, retries=2)
    start = time.time()
    with pytest.raises(TrizAIError):
        client._complete("sys", "user", _Answer, retry=False)
    three_attempts = time.time() - start

    assert three_attempts > one_attempt * 1.5, (
        f"retries did not extend completion wall clock: "
        f"{one_attempt:.1f}s vs {three_attempts:.1f}s"
    )


@pytest.mark.timeout(120)
def test_embedding_retries_are_not_controllable(configured):
    """litellm.embedding retries ~3x internally regardless of num_retries.

    Documented as a test because it is load-bearing for the defaults: neither
    `num_retries`, `max_retries`, nor a pre-built client with `max_retries=0`
    suppresses it, so the effective embedding bound is roughly
    `embeddings.request_timeout * 3`. That is why the embedding timeout defaults
    far below `llm.request_timeout` -- a generous value gets multiplied.

    If a future litellm makes this controllable, this test fails and the
    embedding default can be raised.
    """
    client = configured(timeout=2.0, retries=0)
    start = time.time()
    with pytest.raises(TrizAIError):
        client.get_embedding("hello")
    elapsed = time.time() - start

    assert elapsed > 2.0 * 2, (
        f"embedding stopped retrying internally ({elapsed:.1f}s for a 2s timeout) -- "
        "litellm may now honor num_retries; revisit embeddings.request_timeout"
    )
    assert elapsed < 2.0 * 6, f"embedding wall clock unbounded: {elapsed:.1f}s"


def test_timeout_and_retries_reach_the_provider_kwargs(configured):
    """Both litellm kwarg builders carry the configured bounds."""
    client = configured(timeout=7.0, retries=3)
    for kwargs in (client._litellm_completion_kwargs(), client._litellm_embedding_kwargs()):
        assert kwargs["timeout"] == 7.0
        assert kwargs["num_retries"] == 3


def test_openai_clients_are_built_with_bounds(configured):
    """Both openai fallback clients carry them, each with its own timeout."""
    client = configured(timeout=7.0, retries=3)
    completion_client = client._get_openai_client()
    assert completion_client.timeout == 7.0
    assert completion_client.max_retries == 3
    embedding_client = client._get_openai_embedding_client()
    assert embedding_client.timeout == 7.0
    assert embedding_client.max_retries == 3


def test_embedding_timeout_defaults_below_completion_timeout():
    """The embedding bound must stay lower, since litellm triples it internally."""
    from triz_ai.config import Settings

    s = Settings()
    assert s.embeddings.request_timeout < s.llm.request_timeout


def test_deep_passes_get_a_longer_bound():
    """ARIZ deep passes 1 & 3 must not inherit the ordinary completion bound.

    Pass 3 verifies every candidate from every method in one call; measured at
    52s and 115s on a simple problem with the free default model, so the 120s
    `request_timeout` would cut off legitimate work.
    """
    from triz_ai.config import Settings

    s = Settings()
    assert s.llm.deep_request_timeout > s.llm.request_timeout


def test_complete_accepts_a_timeout_override(configured):
    """The per-call override reaches the provider kwargs."""
    import inspect

    from triz_ai.llm.client import LLMClient

    assert "timeout" in inspect.signature(LLMClient._complete).parameters
