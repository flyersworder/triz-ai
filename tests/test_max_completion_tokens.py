"""The output cap must go on the wire as `max_completion_tokens`, never `max_tokens`.

Regression guard for #53: newer OpenAI reasoning models reject `max_tokens` with a
400 ("Use 'max_completion_tokens' instead"). Gateways only rewrite the parameter
for models they already recognise, so a brand-new model name received it as sent.
Both backends were affected -- the openai SDK passes kwargs through untouched, and
litellm forwards `max_tokens` verbatim for openai/azure models not in its registry.
"""

import pytest
from pydantic import BaseModel

from triz_ai.llm.client import HAS_LITELLM, LLMClient

requires_litellm = pytest.mark.skipif(not HAS_LITELLM, reason="litellm extra not installed")


class _Answer(BaseModel):
    answer: str


class _Msg:
    content = '{"answer": "ok"}'


class _Choice:
    finish_reason = "stop"
    message = _Msg()


class _Response:
    choices = [_Choice()]


@pytest.fixture
def client(monkeypatch, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm:\n  api_base: http://127.0.0.1:9/v1\n  api_key: sk-test\n")
    monkeypatch.setenv("TRIZ_AI_CONFIG", str(cfg))
    return LLMClient()


def _capture_openai_sdk(monkeypatch, client):
    seen: dict = {}

    class _Completions:
        def create(self, **kwargs):
            seen.update(kwargs)
            return _Response()

    class _Chat:
        completions = _Completions()

    class _FakeOpenAI:
        chat = _Chat()

    monkeypatch.setattr("triz_ai.llm.client.HAS_LITELLM", False)
    monkeypatch.setattr(client, "_get_openai_client", lambda: _FakeOpenAI())
    return seen


def _capture_litellm(monkeypatch):
    seen: dict = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return _Response()

    monkeypatch.setattr("triz_ai.llm.client.litellm.completion", fake_completion)
    return seen


def test_openai_sdk_sends_max_completion_tokens(monkeypatch, client):
    seen = _capture_openai_sdk(monkeypatch, client)
    client._complete("sys", "user", _Answer, max_tokens=1024)
    assert seen.get("max_completion_tokens") == 1024
    assert "max_tokens" not in seen


@requires_litellm
def test_litellm_sends_max_completion_tokens(monkeypatch, client):
    seen = _capture_litellm(monkeypatch)
    client._complete("sys", "user", _Answer, max_tokens=1024)
    assert seen.get("max_completion_tokens") == 1024
    assert "max_tokens" not in seen


def test_no_cap_sends_neither_parameter(monkeypatch, client):
    seen = _capture_openai_sdk(monkeypatch, client)
    client._complete("sys", "user", _Answer)
    assert "max_completion_tokens" not in seen
    assert "max_tokens" not in seen


@requires_litellm
@pytest.mark.parametrize(
    ("model", "provider", "wire_key"),
    [
        # The #53 case: an OpenAI model litellm does not know yet.
        ("gpt-6-luna", "azure", "max_completion_tokens"),
        ("gpt-6-luna", "openai", "max_completion_tokens"),
        # The cap on classify calls must survive translation to other providers,
        # or a 1024-token call reserves the model's whole output window.
        ("nvidia/nemotron-3-super-120b-a12b:free", "openrouter", "max_completion_tokens"),
        ("claude-sonnet-4-5", "anthropic", "max_tokens"),
        ("gemini-2.5-flash", "gemini", "max_output_tokens"),
    ],
)
def test_litellm_translates_the_cap_per_provider(model, provider, wire_key):
    from litellm.utils import get_optional_params

    params = get_optional_params(
        model=model, custom_llm_provider=provider, max_completion_tokens=1024
    )
    assert params.get(wire_key) == 1024
    if wire_key != "max_tokens":
        # Both on the wire would still trip the #53 rejection.
        assert "max_tokens" not in params
