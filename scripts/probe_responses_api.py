"""Probe: compare chat.completions vs Responses API for a single TRIZ call.

Issue #18 spike. Picks the simplest pydantic schema in the codebase
(ProblemClassification, used by LLMClient.classify_problem) and runs the
same logical request three ways:

  A. litellm.completion(..., response_format={"type": "json_object"})
     — mirrors current LLMClient._complete()
  B. litellm.responses(..., text={"format": {"type": "json_object"}})
     — Responses API, loose JSON-object mode
  C. litellm.responses(..., text={"format": {"type": "json_schema",
                                              "schema": <model_json_schema>}})
     — Responses API, strict server-enforced schema

For each path: capture latency, raw response shape, JSON-validity (pydantic
round-trip), token usage, and errors. Print a comparison table.

Run:
    uv run python scripts/probe_responses_api.py
    uv run python scripts/probe_responses_api.py \\
        --model openrouter/google/gemini-3.1-flash-lite-preview
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

from triz_ai.engine.ariz import StructuredProblemModel
from triz_ai.llm.client import ProblemClassification
from triz_ai.llm.prompts import classify_problem_prompt, deep_reformulation_prompt

load_dotenv()
litellm.suppress_debug_info = True  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


TEST_PROBLEM = (
    "Our SiC MOSFET gate driver emits enough conducted EMI to fail CISPR 25 Class 5, "
    "but slowing the gate edges to attenuate the EMI bumps switching losses past our "
    "thermal budget. We need to reduce EMI without sacrificing efficiency."
)


# Map of --schema arg → (response_model, system_prompt_fn)
SCHEMAS: dict[str, tuple[type[BaseModel], Any]] = {
    "problem_classification": (ProblemClassification, classify_problem_prompt),
    "structured_problem": (StructuredProblemModel, lambda: deep_reformulation_prompt()),
}


def strictify_schema(schema: dict) -> dict:
    """Transform a pydantic JSON Schema into OpenAI strict-mode shape.

    Rules (per OpenAI Structured Outputs spec):
      - Every object must have `additionalProperties: false`
      - Every property in `properties` must appear in `required`
      - `default` keys are not allowed; strip them
      - Optional pydantic fields (`anyOf` of T and null) stay as-is — strict
        mode supports nullable via the type union, but `default: null` must go
      - `$defs` and `$ref` are supported as of GA structured outputs

    This walks the whole tree mutating in place, then returns the same dict.
    """

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                props = node.get("properties")
                if isinstance(props, dict) and props:
                    node["required"] = list(props.keys())
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(schema)
    # Walk $defs separately to ensure inner object definitions are also strictified.
    for d in schema.get("$defs", {}).values():
        _walk(d)
    return schema


@dataclass
class ProbeResult:
    path: str
    model: str
    ok: bool
    latency_s: float
    raw_text: str | None = None
    parsed: BaseModel | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def _extract_responses_text(response: Any) -> str:
    """Pull the assistant text out of a Responses-API result.

    Responses API returns `output` as a list of items; text lives in
    items of type 'message' whose content has type 'output_text'.
    `output_text` is a SDK convenience accessor when present.
    """
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    out = getattr(response, "output", None) or response.get("output", [])
    chunks: list[str] = []
    for item in out:
        item_type = getattr(item, "type", None) or item.get("type")
        if item_type != "message":
            continue
        content = getattr(item, "content", None) or item.get("content", [])
        for c in content:
            c_type = getattr(c, "type", None) or c.get("type")
            if c_type in ("output_text", "text"):
                chunks.append(getattr(c, "text", None) or c.get("text", ""))
    return "\n".join(chunks)


def _usage_dict(response: Any) -> dict[str, Any] | None:
    u = getattr(response, "usage", None)
    if u is None:
        return None
    if hasattr(u, "model_dump"):
        return u.model_dump()
    if hasattr(u, "to_dict"):
        return u.to_dict()
    if hasattr(u, "__dict__"):
        return {k: v for k, v in u.__dict__.items() if not k.startswith("_")}
    try:
        return dict(u)
    except Exception:
        return {"raw": repr(u)}


def probe_chat_completions(
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_model: type[BaseModel],
    max_tokens: int,
) -> ProbeResult:
    path = "A: chat.completions (json_object)"
    t0 = time.monotonic()
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
        )
        latency = time.monotonic() - t0
        raw = response.choices[0].message.content
        parsed = response_model.model_validate(json.loads(raw))
        return ProbeResult(
            path=path,
            model=model,
            ok=True,
            latency_s=latency,
            raw_text=raw,
            parsed=parsed,
            usage=_usage_dict(response),
        )
    except Exception as e:
        return ProbeResult(
            path=path,
            model=model,
            ok=False,
            latency_s=time.monotonic() - t0,
            error=f"{type(e).__name__}: {e}",
            notes=[traceback.format_exc(limit=2)],
        )


def probe_chat_completions_json_schema(
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_model: type[BaseModel],
    max_tokens: int,
) -> ProbeResult:
    """Path D: chat.completions with response_format=json_schema (strict).

    This is the option we never probed: keep the chat.completions API surface
    (LiteLLM-portable across providers) but upgrade the output contract from
    loose `json_object` to strict `json_schema`. Available in OpenAI's
    Structured Outputs (GA Aug 2024) and translated to provider-native
    mechanisms (Anthropic tool-use, etc.) by LiteLLM.
    """
    path = "D: chat.completions (json_schema strict)"
    schema = strictify_schema(response_model.model_json_schema())
    t0 = time.monotonic()
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": schema,
                    "strict": True,
                },
            },
            max_tokens=max_tokens,
        )
        latency = time.monotonic() - t0
        raw = response.choices[0].message.content
        parsed = response_model.model_validate(json.loads(raw))
        return ProbeResult(
            path=path,
            model=model,
            ok=True,
            latency_s=latency,
            raw_text=raw,
            parsed=parsed,
            usage=_usage_dict(response),
        )
    except Exception as e:
        return ProbeResult(
            path=path,
            model=model,
            ok=False,
            latency_s=time.monotonic() - t0,
            error=f"{type(e).__name__}: {e}",
            notes=[traceback.format_exc(limit=2)],
        )


def probe_responses_json_object(
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_model: type[BaseModel],
    max_tokens: int,
) -> ProbeResult:
    path = "B: responses (json_object)"
    t0 = time.monotonic()
    try:
        response = litellm.responses(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={"format": {"type": "json_object"}},
            max_output_tokens=max_tokens,
        )
        latency = time.monotonic() - t0
        raw = _extract_responses_text(response)
        parsed = response_model.model_validate(json.loads(raw))
        return ProbeResult(
            path=path,
            model=model,
            ok=True,
            latency_s=latency,
            raw_text=raw,
            parsed=parsed,
            usage=_usage_dict(response),
        )
    except Exception as e:
        return ProbeResult(
            path=path,
            model=model,
            ok=False,
            latency_s=time.monotonic() - t0,
            error=f"{type(e).__name__}: {e}",
            notes=[traceback.format_exc(limit=2)],
        )


def probe_responses_json_schema(
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_model: type[BaseModel],
    max_tokens: int,
) -> ProbeResult:
    path = "C: responses (json_schema strict)"
    schema = strictify_schema(response_model.model_json_schema())
    t0 = time.monotonic()
    try:
        response = litellm.responses(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": response_model.__name__,
                    "schema": schema,
                    "strict": True,
                }
            },
            max_output_tokens=max_tokens,
        )
        latency = time.monotonic() - t0
        raw = _extract_responses_text(response)
        parsed = response_model.model_validate(json.loads(raw))
        return ProbeResult(
            path=path,
            model=model,
            ok=True,
            latency_s=latency,
            raw_text=raw,
            parsed=parsed,
            usage=_usage_dict(response),
        )
    except Exception as e:
        return ProbeResult(
            path=path,
            model=model,
            ok=False,
            latency_s=time.monotonic() - t0,
            error=f"{type(e).__name__}: {e}",
            notes=[traceback.format_exc(limit=2)],
        )


def _summarize_parsed(parsed: BaseModel) -> str:
    """Short human-readable summary of a parsed response, schema-aware."""
    if isinstance(parsed, ProblemClassification):
        return (
            f"primary={parsed.primary_method!r}, "
            f"secondary={parsed.secondary_method!r}, conf={parsed.confidence}\n"
            f"                reformulated: {parsed.reformulated_problem[:80]}…"
        )
    if isinstance(parsed, StructuredProblemModel):
        tc1 = parsed.technical_contradiction_1
        pc = parsed.physical_contradiction
        ri = parsed.resource_inventory
        return (
            f"reformulated: {parsed.reformulated_problem[:70]}…\n"
            f"                TC1: improve {tc1.improving_param_id} "
            f"({tc1.improving_param_name}) vs worsen {tc1.worsening_param_id} "
            f"({tc1.worsening_param_name})\n"
            f"                PC: {'present' if pc else 'None'} | "
            f"resources: {len(ri.substances)} substances, {len(ri.fields)} fields\n"
            f"                tools: {parsed.recommended_tools}"
        )
    return str(parsed)[:200]


def render(results: list[ProbeResult], schema_name: str) -> None:
    print()
    print("=" * 100)
    print(f"PROBE RESULTS  schema={schema_name}  problem={TEST_PROBLEM[:50]}…")
    print("=" * 100)
    for r in results:
        status = "OK " if r.ok else "FAIL"
        print(f"\n[{status}] {r.path}")
        print(f"      model:    {r.model}")
        print(f"      latency:  {r.latency_s:.2f}s")
        if r.usage:
            print(f"      usage:    {r.usage}")
        if r.ok and r.parsed:
            print(f"      parsed:   {_summarize_parsed(r.parsed)}")
        if not r.ok:
            print(f"      error:    {r.error}")
            if r.raw_text:
                print(f"      raw:      {r.raw_text[:400]}")
    print("\n" + "=" * 100)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="openrouter/nvidia/nemotron-3-nano-30b-a3b:free",
        help="Model to probe (default: triz-ai's classify_model)",
    )
    parser.add_argument(
        "--paths",
        default="abcd",
        help="Which paths to run, e.g. 'ad' or 'c'. Default: all four (a, b, c, d).",
    )
    parser.add_argument(
        "--schema",
        choices=sorted(SCHEMAS.keys()),
        default="problem_classification",
        help="Which pydantic response model + prompt to use.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="max_tokens override (default: 1024 for simple, 4096 for nested schemas).",
    )
    args = parser.parse_args()

    response_model, system_prompt_fn = SCHEMAS[args.schema]
    system_prompt = system_prompt_fn()
    user_prompt = TEST_PROBLEM
    max_tokens = args.max_tokens or (4096 if args.schema in {"structured_problem"} else 1024)

    results: list[ProbeResult] = []
    if "a" in args.paths.lower():
        print(f"→ Path A: chat.completions on {args.model} [{args.schema}]")
        results.append(
            probe_chat_completions(
                args.model, system_prompt, user_prompt, response_model, max_tokens
            )
        )
    if "b" in args.paths.lower():
        print(f"→ Path B: responses (json_object) on {args.model} [{args.schema}]")
        results.append(
            probe_responses_json_object(
                args.model, system_prompt, user_prompt, response_model, max_tokens
            )
        )
    if "c" in args.paths.lower():
        print(f"→ Path C: responses (json_schema strict) on {args.model} [{args.schema}]")
        results.append(
            probe_responses_json_schema(
                args.model, system_prompt, user_prompt, response_model, max_tokens
            )
        )
    if "d" in args.paths.lower():
        print(f"→ Path D: chat.completions (json_schema strict) on {args.model} [{args.schema}]")
        results.append(
            probe_chat_completions_json_schema(
                args.model, system_prompt, user_prompt, response_model, max_tokens
            )
        )

    render(results, args.schema)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
