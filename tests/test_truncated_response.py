"""Output-budget overruns must be legible and must not be retried.

Regression guard for the deep-mode pass 3 failure: `max_tokens` was hard-coded
to 6144 while the call genuinely needs 8.5k-10.5k completion tokens. Because the
model reasons inside the content field, it spent the whole budget before emitting
any JSON -- so the overrun surfaced as `Expecting value: line 1 column 1 (char 0)`,
which points at malformed JSON rather than at a token limit, and the application
retry then burned a second 70-200s call on a guaranteed failure.
"""

import pytest
from pydantic import BaseModel

from triz_ai.config import Settings
from triz_ai.llm.client import (
    HAS_LITELLM,
    LLMClient,
    TrizAIError,
    TruncatedResponseError,
    _is_retryable,
    _raise_if_truncated,
)

requires_litellm = pytest.mark.skipif(not HAS_LITELLM, reason="litellm extra not installed")


class _Answer(BaseModel):
    answer: str


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, finish_reason, content):
        self.finish_reason = finish_reason
        self.message = _Msg(content)


class _Usage:
    def __init__(self, completion_tokens):
        self.completion_tokens = completion_tokens


class _Response:
    def __init__(self, finish_reason, content="", completion_tokens=6144):
        self.choices = [_Choice(finish_reason, content)]
        self.usage = _Usage(completion_tokens)


def test_truncation_raises_instead_of_reaching_the_json_parser():
    with pytest.raises(TruncatedResponseError):
        _raise_if_truncated(_Response("length"), "some/model", 6144)


def test_normal_finish_is_left_alone():
    _raise_if_truncated(_Response("stop", '{"answer": "ok"}'), "some/model", 6144)


def test_error_names_the_budget_and_the_config_key():
    """The whole point is that the message explains the real problem."""
    with pytest.raises(TruncatedResponseError) as exc:
        _raise_if_truncated(_Response("length", completion_tokens=6144), "some/model", 6144)
    msg = str(exc.value)
    assert "6144" in msg
    assert "llm.deep_max_output_tokens" in msg
    # The reason it masquerades as a parse error is worth stating in the message.
    assert "reasoning" in msg.lower()


def test_truncation_is_not_retryable():
    """Retrying under the same budget overruns it again, at 70-200s per attempt."""
    assert _is_retryable(TruncatedResponseError("x")) is False


@requires_litellm
def test_complete_surfaces_truncation_without_retrying(monkeypatch, tmp_path):
    """End to end through _complete: one call, and a legible error."""
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _Response("length", "We need to verify each candidate...", 6144)

    monkeypatch.setattr("triz_ai.llm.client.litellm.completion", fake_completion)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm:\n  api_base: http://127.0.0.1:9/v1\n  api_key: sk-test\n")
    monkeypatch.setenv("TRIZ_AI_CONFIG", str(cfg))

    client = LLMClient()
    with pytest.raises(TrizAIError) as exc:
        client._complete("sys", "user", _Answer, retry=True, max_tokens=6144)

    assert len(calls) == 1, f"truncation was retried {len(calls)} times; it cannot succeed"
    assert "output budget" in str(exc.value)


def test_pass3_budget_is_configurable_and_exceeds_the_measured_need():
    """Pass 3 measured 8.5k-10.5k completion tokens; the default must clear that."""
    assert Settings().llm.deep_max_output_tokens >= 12000


@requires_litellm
def test_verify_and_synthesize_uses_the_configured_budget(monkeypatch, tmp_path):
    """The hard-coded 6144 is gone -- pass 3 reads the config value."""
    seen = {}

    def fake_complete(self, *args, **kwargs):
        seen["max_tokens"] = kwargs.get("max_tokens")
        raise TrizAIError("stop here")

    monkeypatch.setattr(LLMClient, "_complete", fake_complete)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm:\n  deep_max_output_tokens: 20000\n")
    monkeypatch.setenv("TRIZ_AI_CONFIG", str(cfg))

    from triz_ai.engine.ariz import (
        ResourceInventory,
        StructuredProblemModel,
        TechnicalContradiction,
    )

    tc = TechnicalContradiction(
        improving_param_id=1,
        improving_param_name="Weight of moving object",
        worsening_param_id=2,
        worsening_param_name="Weight of stationary object",
        intensified_description="d",
    )
    problem = StructuredProblemModel(
        original_problem="p",
        reformulated_problem="p",
        technical_contradiction_1=tc,
        technical_contradiction_2=tc,
        physical_contradiction=None,
        ideal_final_result="ifr",
        resource_inventory=ResourceInventory(
            substances=[], fields=[], time_resources=[], space_resources=[]
        ),
        recommended_tools=[],
        recommended_research_tools=[],
        reasoning="r",
    )
    client = LLMClient()
    with pytest.raises(TrizAIError):
        client.verify_and_synthesize(problem, [])
    assert seen["max_tokens"] == 20000
