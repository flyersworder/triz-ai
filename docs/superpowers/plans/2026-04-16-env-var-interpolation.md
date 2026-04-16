# Config Env Var Interpolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support `${VAR}` and `${VAR:-default}` interpolation in the YAML config, resolved once in `load_config()` via a hand-written scanner, so K8s/OpenShift deployments can inject API keys from Secrets without hardcoding them in the config or the Docker image.

**Architecture:** Pure preprocessing layer. After `yaml.safe_load()` parses the YAML into a `dict`, a new `_interpolate_env(data)` walks it recursively and rewrites every string leaf by delegating to `_resolve_tokens(value, field_path)` — a char-by-char scanner that handles `$$` escapes, `${VAR}`, and `${VAR:-default}`. Pydantic models, `LLMClient`, CLI, and existing tests are untouched. Unset/empty `${VAR}` (no default) raises `ConfigError` at startup, naming the exact field (`llm.api_key`, `database.vector_options[0].endpoint`).

**Tech Stack:** Python 3.12+, pydantic/pydantic-settings (already used), pytest with `monkeypatch` + `tmp_path`. No new dependencies.

**Spec:** `docs/specs/2026-04-16-env-var-interpolation-design.md`

**Issue:** [#9](https://github.com/flyersworder/triz-ai/issues/9)

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `src/triz_ai/config.py` | Modify | Add `ConfigError`, `_resolve_tokens`, `_interpolate_env`; call `_interpolate_env(data)` in `load_config()` between `yaml.safe_load()` and `Settings(**data)` |
| `tests/test_config_interpolation.py` | **Create** | Unit tests for scanner + walker; integration tests through `load_config()` |
| `README.md` | Modify | Add "Environment variable interpolation" subsection after the existing `~/.triz-ai/config.yaml` documentation (around line 268) |
| `CLAUDE.md` | Modify | Add one-line note under References about interpolation syntax |

---

### Task 1: Core scanner — `${VAR}` + escape + multiple tokens

This task scaffolds `ConfigError` and implements the happy paths of `_resolve_tokens`: plain string pass-through, single `${VAR}` substitution, `$$` escape, and multiple tokens in one string. Error cases and defaults are added in later tasks.

**Files:**
- Modify: `src/triz_ai/config.py`
- Create: `tests/test_config_interpolation.py`

- [ ] **Step 1: Write failing tests for scanner happy paths**

Create `tests/test_config_interpolation.py`:

```python
"""Tests for YAML config environment variable interpolation."""

import pytest

from triz_ai.config import ConfigError, _resolve_tokens


def test_plain_string_passthrough():
    assert _resolve_tokens("hello world", "f") == "hello world"


def test_empty_string_passthrough():
    assert _resolve_tokens("", "f") == ""


def test_single_var_resolves(monkeypatch):
    monkeypatch.setenv("TRIZ_TEST_TOKEN", "hello")
    assert _resolve_tokens("${TRIZ_TEST_TOKEN}", "f") == "hello"


def test_var_embedded_in_string(monkeypatch):
    monkeypatch.setenv("TRIZ_HOST", "db.internal")
    assert _resolve_tokens("http://${TRIZ_HOST}/v1", "f") == "http://db.internal/v1"


def test_multiple_tokens(monkeypatch):
    monkeypatch.setenv("TRIZ_A", "x")
    monkeypatch.setenv("TRIZ_B", "y")
    assert _resolve_tokens("${TRIZ_A}-${TRIZ_B}", "f") == "x-y"


def test_escape_dollar():
    assert _resolve_tokens("$$", "f") == "$"


def test_escape_before_brace():
    assert _resolve_tokens("$${FOO}", "f") == "${FOO}"


def test_multiple_escapes():
    assert _resolve_tokens("$$$${FOO}", "f") == "$${FOO}"


def test_escape_and_var_mixed(monkeypatch):
    monkeypatch.setenv("TRIZ_FOO", "x")
    assert _resolve_tokens("${TRIZ_FOO}$${BAR}", "f") == "x${BAR}"


def test_single_dollar_not_at_token_start():
    # A lone '$' not followed by '$' or '{' passes through literally
    assert _resolve_tokens("price: $5", "f") == "price: $5"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config_interpolation.py -v`

Expected: All tests FAIL at import time with `ImportError: cannot import name 'ConfigError' from 'triz_ai.config'` (or `_resolve_tokens` — whichever the import resolves first).

- [ ] **Step 3: Add `ConfigError` and minimal `_resolve_tokens` to `src/triz_ai/config.py`**

Open `src/triz_ai/config.py`. After the existing `load_dotenv()` call (line 11) and before `class LLMConfig`, insert:

```python
class ConfigError(Exception):
    """Raised when config loading or env var interpolation fails."""


_NAME_START = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_"
_NAME_CONT = _NAME_START + "0123456789"


def _resolve_tokens(value: str, field_path: str) -> str:
    """Resolve ${VAR} and ${VAR:-default} tokens in a string value.

    See docs/specs/2026-04-16-env-var-interpolation-design.md for the grammar.
    Defaults and error cases are added in later tasks.
    """
    out: list[str] = []
    i = 0
    n = len(value)
    while i < n:
        if value[i] == "$" and i + 1 < n and value[i + 1] == "$":
            out.append("$")
            i += 2
            continue
        if value[i] == "$" and i + 1 < n and value[i + 1] == "{":
            i += 2  # past '${'
            name_start = i
            if i < n and value[i] in _NAME_START:
                i += 1
                while i < n and value[i] in _NAME_CONT:
                    i += 1
            name = value[name_start:i]
            # Closing '}' expected (defaults added in Task 2)
            i += 1  # past '}'
            env_val = os.environ.get(name)
            out.append(env_val or "")
            continue
        out.append(value[i])
        i += 1
    return "".join(out)
```

Note: `os` is already imported at the top of `config.py` (line 3).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_interpolation.py -v`

Expected: All 10 tests PASS. The scanner currently returns `""` for unset vars (temporary — tightened in Task 3), but none of these tests exercise the unset case yet.

- [ ] **Step 5: Run type check**

Run: `uvx ty check src/triz_ai/config.py`

Expected: No new errors. (Pre-existing project-wide warnings are unchanged.)

- [ ] **Step 6: Commit**

```bash
git add src/triz_ai/config.py tests/test_config_interpolation.py
git commit -m "feat(config): scaffold env var interpolation scanner (#9)

Adds ConfigError and _resolve_tokens handling \${VAR}, \$\$ escape,
and multiple tokens per string. Defaults and error cases follow in
subsequent commits."
```

---

### Task 2: Scanner — `${VAR:-default}` syntax

Extend the scanner to handle `${VAR:-default}`, using shell `:-` semantics where both unset AND empty-string env vars fall back to `default`. `${VAR:-}` yields the empty string. Defaults are read verbatim (no re-interpolation).

**Files:**
- Modify: `src/triz_ai/config.py`
- Modify: `tests/test_config_interpolation.py`

- [ ] **Step 1: Add failing tests for defaults**

Append to `tests/test_config_interpolation.py`:

```python
def test_default_used_when_unset(monkeypatch):
    monkeypatch.delenv("TRIZ_NOT_SET", raising=False)
    assert _resolve_tokens("${TRIZ_NOT_SET:-fallback}", "f") == "fallback"


def test_default_used_when_empty(monkeypatch):
    monkeypatch.setenv("TRIZ_EMPTY", "")
    assert _resolve_tokens("${TRIZ_EMPTY:-fallback}", "f") == "fallback"


def test_default_ignored_when_var_set(monkeypatch):
    monkeypatch.setenv("TRIZ_HOST", "prod")
    assert _resolve_tokens("${TRIZ_HOST:-default}", "f") == "prod"


def test_empty_default(monkeypatch):
    monkeypatch.delenv("TRIZ_MAYBE", raising=False)
    assert _resolve_tokens("${TRIZ_MAYBE:-}", "f") == ""


def test_default_with_colons_dashes_spaces(monkeypatch):
    monkeypatch.delenv("TRIZ_URL", raising=False)
    result = _resolve_tokens("${TRIZ_URL:-https://api.example.com:443/v1}", "f")
    assert result == "https://api.example.com:443/v1"


def test_default_not_reinterpolated(monkeypatch):
    # Default value is a literal; it is NOT itself re-interpolated
    monkeypatch.delenv("TRIZ_OUTER", raising=False)
    monkeypatch.setenv("TRIZ_INNER", "secret")
    assert _resolve_tokens("${TRIZ_OUTER:-${TRIZ_INNER}}", "f") == "${TRIZ_INNER}"


def test_env_value_not_reinterpolated(monkeypatch):
    # Single-pass: an env value that happens to look like ${...} is verbatim
    monkeypatch.setenv("TRIZ_OUTER", "${TRIZ_INNER}")
    monkeypatch.setenv("TRIZ_INNER", "should_not_be_seen")
    assert _resolve_tokens("${TRIZ_OUTER}", "f") == "${TRIZ_INNER}"
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/test_config_interpolation.py -v -k "default or reinterpolated"`

Expected: Several FAIL. Most notably `test_default_used_when_unset` fails because the current scanner consumes up to `}` greedily and interprets `VAR:-fallback` as the whole name, returning empty string. `test_default_not_reinterpolated` fails because the greedy `}` match stops at the first brace, which is inside the default.

- [ ] **Step 3: Extend `_resolve_tokens` to handle `:-default`**

In `src/triz_ai/config.py`, replace the body of `_resolve_tokens` with:

```python
def _resolve_tokens(value: str, field_path: str) -> str:
    """Resolve ${VAR} and ${VAR:-default} tokens in a string value.

    See docs/specs/2026-04-16-env-var-interpolation-design.md for the grammar.
    Error cases are added in Task 3.
    """
    out: list[str] = []
    i = 0
    n = len(value)
    while i < n:
        if value[i] == "$" and i + 1 < n and value[i + 1] == "$":
            out.append("$")
            i += 2
            continue
        if value[i] == "$" and i + 1 < n and value[i + 1] == "{":
            i += 2  # past '${'
            name_start = i
            if i < n and value[i] in _NAME_START:
                i += 1
                while i < n and value[i] in _NAME_CONT:
                    i += 1
            name = value[name_start:i]
            default: str | None = None
            if i < n and value[i] == "}":
                i += 1  # past '}'
            elif i + 1 < n and value[i] == ":" and value[i + 1] == "-":
                i += 2  # past ':-'
                default_end = value.find("}", i)
                # Task 3 turns default_end == -1 into a proper error
                default = value[i:default_end] if default_end != -1 else value[i:]
                i = default_end + 1 if default_end != -1 else n
            else:
                # Task 3 handles unclosed / invalid-char cases
                i += 1
            env_val = os.environ.get(name)
            if env_val:
                out.append(env_val)
            elif default is not None:
                out.append(default)
            else:
                out.append("")  # Task 3 turns this into ConfigError
            continue
        out.append(value[i])
        i += 1
    return "".join(out)
```

- [ ] **Step 4: Run all tests to verify defaults pass**

Run: `uv run pytest tests/test_config_interpolation.py -v`

Expected: All 17 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/triz_ai/config.py tests/test_config_interpolation.py
git commit -m "feat(config): add \${VAR:-default} syntax to scanner (#9)

Shell :- semantics — both unset and empty env vars fall back to
default. Defaults are read verbatim and not re-interpolated."
```

---

### Task 3: Scanner — error cases

Tighten the scanner: unset/empty `${VAR}` (no default) raises `ConfigError` with field path and var name; unclosed `${...`, empty name `${}` / `${:-x}`, and invalid characters inside a name (the nested-token case) also raise.

**Files:**
- Modify: `src/triz_ai/config.py`
- Modify: `tests/test_config_interpolation.py`

- [ ] **Step 1: Add failing tests for error cases**

Append to `tests/test_config_interpolation.py`:

```python
def test_unset_var_raises(monkeypatch):
    monkeypatch.delenv("TRIZ_MISSING", raising=False)
    with pytest.raises(ConfigError) as exc_info:
        _resolve_tokens("${TRIZ_MISSING}", "llm.api_key")
    msg = str(exc_info.value)
    assert "llm.api_key" in msg
    assert "TRIZ_MISSING" in msg


def test_empty_var_raises(monkeypatch):
    monkeypatch.setenv("TRIZ_EMPTY", "")
    with pytest.raises(ConfigError) as exc_info:
        _resolve_tokens("${TRIZ_EMPTY}", "llm.api_key")
    assert "TRIZ_EMPTY" in str(exc_info.value)


def test_error_message_suggests_default_syntax(monkeypatch):
    monkeypatch.delenv("TRIZ_MISSING", raising=False)
    with pytest.raises(ConfigError, match=r"\$\{TRIZ_MISSING:-\}"):
        _resolve_tokens("${TRIZ_MISSING}", "f")


def test_unclosed_token_raises():
    with pytest.raises(ConfigError, match="unclosed"):
        _resolve_tokens("${FOO", "f")


def test_unclosed_token_with_default_raises():
    with pytest.raises(ConfigError, match="unclosed"):
        _resolve_tokens("${FOO:-default_but_no_brace", "f")


def test_empty_name_raises():
    with pytest.raises(ConfigError, match="empty variable name"):
        _resolve_tokens("${}", "f")


def test_empty_name_with_default_raises():
    with pytest.raises(ConfigError, match="empty variable name"):
        _resolve_tokens("${:-x}", "f")


def test_nested_token_raises_invalid_char():
    # ${FOO_${BAR}} — after parsing name "FOO_", the next char is '$'
    # which is not a valid name continuation char and not '}' or ':-'
    with pytest.raises(ConfigError, match="invalid character"):
        _resolve_tokens("${FOO_${BAR}}", "f")


def test_digit_first_char_raises():
    # Names must start with letter/underscore
    with pytest.raises(ConfigError, match="invalid character"):
        _resolve_tokens("${1FOO}", "f")


def test_lone_trailing_dollar_passes_through():
    # Trailing '$' with nothing after it is just a literal '$'
    assert _resolve_tokens("cost: $", "f") == "cost: $"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config_interpolation.py -v -k "raises or lone_trailing"`

Expected: Most FAIL — the Task 2 scanner silently returns empty strings or passes through malformed tokens.

- [ ] **Step 3: Rewrite `_resolve_tokens` with full error handling**

In `src/triz_ai/config.py`, replace the body of `_resolve_tokens` with:

```python
def _resolve_tokens(value: str, field_path: str) -> str:
    """Resolve ${VAR} and ${VAR:-default} tokens in a string value.

    See docs/specs/2026-04-16-env-var-interpolation-design.md for the grammar.
    Raises ConfigError on unclosed tokens, empty/invalid names, or
    unset/empty vars without defaults.
    """
    out: list[str] = []
    i = 0
    n = len(value)
    while i < n:
        # Escape: $$ -> $
        if value[i] == "$" and i + 1 < n and value[i + 1] == "$":
            out.append("$")
            i += 2
            continue
        # Token: ${...}
        if value[i] == "$" and i + 1 < n and value[i + 1] == "{":
            i += 2  # past '${'
            name_start = i
            if i < n and value[i] in _NAME_START:
                i += 1
                while i < n and value[i] in _NAME_CONT:
                    i += 1
            name = value[name_start:i]
            if not name:
                if i >= n:
                    raise ConfigError(
                        f"Config field {field_path}: unclosed '${{' in value {value!r}"
                    )
                if value[i] == "}" or value[i : i + 2] == ":-":
                    raise ConfigError(
                        f"Config field {field_path}: empty variable name in {value!r}"
                    )
                raise ConfigError(
                    f"Config field {field_path}: invalid character {value[i]!r} "
                    f"in variable name; {value!r}"
                )
            default: str | None = None
            if i >= n:
                raise ConfigError(
                    f"Config field {field_path}: unclosed '${{' in value {value!r}"
                )
            if value[i] == "}":
                i += 1  # past '}'
            elif value[i : i + 2] == ":-":
                i += 2  # past ':-'
                end = value.find("}", i)
                if end == -1:
                    raise ConfigError(
                        f"Config field {field_path}: unclosed '${{' in value {value!r}"
                    )
                default = value[i:end]
                i = end + 1  # past '}'
            else:
                raise ConfigError(
                    f"Config field {field_path}: invalid character {value[i]!r} "
                    f"in variable name; {value!r}"
                )
            env_val = os.environ.get(name)
            if env_val:
                out.append(env_val)
            elif default is not None:
                out.append(default)
            else:
                raise ConfigError(
                    f"Config field {field_path}: environment variable {name} is not set "
                    f"(or is empty). Either set it to a non-empty value, or provide a "
                    f"default: ${{{name}:-}}"
                )
            continue
        out.append(value[i])
        i += 1
    return "".join(out)
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `uv run pytest tests/test_config_interpolation.py -v`

Expected: All tests PASS (happy-path + defaults + error cases).

- [ ] **Step 5: Commit**

```bash
git add src/triz_ai/config.py tests/test_config_interpolation.py
git commit -m "feat(config): fail fast on unset/empty env vars & malformed tokens (#9)

Unset or empty \${VAR} (no default) raises ConfigError naming the
field and variable. Unclosed tokens, empty names, and invalid
characters (nested tokens) also raise. Error message suggests
\${VAR:-} as the explicit opt-in for empty values."
```

---

### Task 4: Tree walker `_interpolate_env` + field-path tracking

Add the recursive walker that applies `_resolve_tokens` to every string leaf in a parsed YAML dict. Tracks a dotted field path (`llm.api_key`, `database.vector_options[0].endpoint`) for use in error messages. Non-string scalars pass through unchanged.

**Files:**
- Modify: `src/triz_ai/config.py`
- Modify: `tests/test_config_interpolation.py`

- [ ] **Step 1: Add failing tests for the walker**

Append to `tests/test_config_interpolation.py`:

```python
from triz_ai.config import _interpolate_env


def test_walk_flat_dict(monkeypatch):
    monkeypatch.setenv("TRIZ_KEY", "sk-abc")
    assert _interpolate_env({"api_key": "${TRIZ_KEY}"}) == {"api_key": "sk-abc"}


def test_walk_nested_dict(monkeypatch):
    monkeypatch.setenv("TRIZ_KEY", "secret")
    data = {"llm": {"api_key": "${TRIZ_KEY}", "model": "gpt-4o"}}
    assert _interpolate_env(data) == {"llm": {"api_key": "secret", "model": "gpt-4o"}}


def test_walk_list_of_strings(monkeypatch):
    monkeypatch.setenv("TRIZ_A", "x")
    assert _interpolate_env(["${TRIZ_A}", "plain"]) == ["x", "plain"]


def test_walk_dict_in_list(monkeypatch):
    monkeypatch.setenv("TRIZ_URL", "https://db.internal")
    data = {"database": {"vector_options": [{"endpoint": "${TRIZ_URL}"}]}}
    result = _interpolate_env(data)
    assert result["database"]["vector_options"][0]["endpoint"] == "https://db.internal"


def test_walk_list_in_dict_in_list(monkeypatch):
    monkeypatch.setenv("TRIZ_X", "x")
    data = [{"items": ["${TRIZ_X}", "y"]}]
    assert _interpolate_env(data) == [{"items": ["x", "y"]}]


def test_walk_non_strings_unchanged():
    data = {
        "int_field": 42,
        "bool_field": True,
        "none_field": None,
        "float_field": 3.14,
        "empty_list": [],
        "empty_dict": {},
    }
    assert _interpolate_env(data) == data


def test_walk_dict_keys_not_interpolated(monkeypatch):
    # Keys are schema — never substitute them even if they look like tokens
    monkeypatch.setenv("TRIZ_SHOULD_NOT_APPEAR", "danger")
    data = {"${TRIZ_SHOULD_NOT_APPEAR}": "value"}
    assert _interpolate_env(data) == {"${TRIZ_SHOULD_NOT_APPEAR}": "value"}


def test_walk_empty_dict_and_list():
    assert _interpolate_env({}) == {}
    assert _interpolate_env([]) == []


def test_walk_error_path_dotted(monkeypatch):
    monkeypatch.delenv("TRIZ_MISSING", raising=False)
    data = {"llm": {"api_key": "${TRIZ_MISSING}"}}
    with pytest.raises(ConfigError, match=r"llm\.api_key"):
        _interpolate_env(data)


def test_walk_error_path_list_index(monkeypatch):
    monkeypatch.delenv("TRIZ_MISSING", raising=False)
    data = {"retries": ["ok", "${TRIZ_MISSING}"]}
    with pytest.raises(ConfigError, match=r"retries\[1\]"):
        _interpolate_env(data)


def test_walk_error_path_combined(monkeypatch):
    monkeypatch.delenv("TRIZ_MISSING", raising=False)
    data = {"database": {"vector_options": [{"endpoint": "${TRIZ_MISSING}"}]}}
    with pytest.raises(ConfigError, match=r"database\.vector_options\[0\]\.endpoint"):
        _interpolate_env(data)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config_interpolation.py -v -k "walk"`

Expected: All FAIL with `ImportError: cannot import name '_interpolate_env' from 'triz_ai.config'`.

- [ ] **Step 3: Implement `_interpolate_env`**

In `src/triz_ai/config.py`, add this function immediately after `_resolve_tokens`:

```python
def _interpolate_env(data, field_path: str = ""):
    """Recursively interpolate env vars in string leaves of a parsed YAML structure.

    Walks dicts and lists; returns non-string, non-container values unchanged.
    Dict keys are never interpolated. `field_path` is accumulated for
    error messages (e.g., 'llm.api_key', 'database.vector_options[0].endpoint').
    """
    if isinstance(data, dict):
        return {
            k: _interpolate_env(v, f"{field_path}.{k}" if field_path else k)
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [_interpolate_env(v, f"{field_path}[{idx}]") for idx, v in enumerate(data)]
    if isinstance(data, str):
        return _resolve_tokens(data, field_path or "<root>")
    return data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_interpolation.py -v`

Expected: All tests PASS (scanner + walker).

- [ ] **Step 5: Commit**

```bash
git add src/triz_ai/config.py tests/test_config_interpolation.py
git commit -m "feat(config): add _interpolate_env walker with field-path tracking (#9)

Recursively applies _resolve_tokens to string leaves in dicts/lists.
Dict keys are never interpolated. Error messages include the dotted
field path (e.g., database.vector_options[0].endpoint)."
```

---

### Task 5: Wire into `load_config()` + integration tests

Plug `_interpolate_env(data)` into `load_config()` between `yaml.safe_load()` and `Settings(**data)`. Add integration tests that exercise the full path: write a YAML file with tokens, set env vars, call `load_config(path)`, assert resolved `Settings`.

**Files:**
- Modify: `src/triz_ai/config.py`
- Modify: `tests/test_config_interpolation.py`

- [ ] **Step 1: Add failing integration tests**

Append to `tests/test_config_interpolation.py`:

```python
def test_load_config_resolves_env_vars(tmp_path, monkeypatch):
    from triz_ai.config import load_config

    monkeypatch.setenv("TRIZ_MY_KEY", "sk-abc")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        'llm:\n'
        '  api_base: "${TRIZ_API_BASE:-https://default.example.com/v1}"\n'
        '  api_key: "${TRIZ_MY_KEY}"\n'
    )
    settings = load_config(cfg)
    assert settings.llm.api_key == "sk-abc"
    assert settings.llm.api_base == "https://default.example.com/v1"


def test_load_config_backward_compat_hardcoded_values(tmp_path):
    from triz_ai.config import load_config

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        'llm:\n'
        '  api_base: "https://gateway.example.com/v1"\n'
        '  api_key: "hardcoded-literal"\n'
    )
    settings = load_config(cfg)
    assert settings.llm.api_key == "hardcoded-literal"
    assert settings.llm.api_base == "https://gateway.example.com/v1"


def test_load_config_missing_env_raises(tmp_path, monkeypatch):
    from triz_ai.config import load_config

    monkeypatch.delenv("TRIZ_SHOULD_EXIST", raising=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text('llm:\n  api_key: "${TRIZ_SHOULD_EXIST}"\n')
    with pytest.raises(ConfigError, match=r"llm\.api_key.*TRIZ_SHOULD_EXIST"):
        load_config(cfg)


def test_load_config_empty_yaml_still_works(tmp_path):
    from triz_ai.config import load_config

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("")
    settings = load_config(cfg)
    # Sanity: defaults still load
    assert settings.embeddings.dimensions == 768


def test_load_config_non_string_fields_unchanged(tmp_path):
    from triz_ai.config import load_config

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("embeddings:\n  dimensions: 768\nevolution:\n  auto_classify: true\n")
    settings = load_config(cfg)
    assert settings.embeddings.dimensions == 768
    assert settings.evolution.auto_classify is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config_interpolation.py -v -k "load_config"`

Expected: `test_load_config_resolves_env_vars` FAILS — `load_config()` currently stores the literal string `"${TRIZ_MY_KEY}"` into `settings.llm.api_key` because no interpolation runs. `test_load_config_missing_env_raises` FAILS for the same reason (no `ConfigError` raised; the literal string is stored). The other three tests should actually PASS already (backward compat, empty YAML, non-string fields).

- [ ] **Step 3: Wire `_interpolate_env` into `load_config`**

In `src/triz_ai/config.py`, find the `load_config` function (around line 63). The existing body of the `if config_path.exists():` branch is:

```python
    if config_path.exists():
        import yaml

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return Settings(**data)
    return Settings()
```

Replace it with:

```python
    if config_path.exists():
        import yaml

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        data = _interpolate_env(data)
        return Settings(**data)
    return Settings()
```

- [ ] **Step 4: Run all config tests to verify they pass**

Run: `uv run pytest tests/test_config_interpolation.py tests/test_config.py -v`

Expected: All tests PASS — new integration tests plus the pre-existing `test_config.py` (which doesn't use a YAML file, so is unaffected).

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run: `uv run pytest`

Expected: All previously passing tests continue to pass. (The new tests in `test_config_interpolation.py` are purely additive.)

- [ ] **Step 6: Commit**

```bash
git add src/triz_ai/config.py tests/test_config_interpolation.py
git commit -m "feat(config): wire env var interpolation into load_config (#9)

load_config() now calls _interpolate_env() between yaml.safe_load
and Settings(**data). Backward compatible: YAML files without
\${...} tokens pass through unchanged. Unset/empty required vars
fail fast at startup with a ConfigError naming the exact field."
```

---

### Task 6: Documentation — README + CLAUDE.md

Document the feature in the two places users and future contributors will look.

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add "Environment variable interpolation" subsection to README**

Open `README.md`. Find the "Using a custom LLM gateway" subsection — it ends around line 268 with a YAML example that hardcodes `api_key: your-proxy-token`. Immediately after that example (after its closing ``` fence, before the "You can also override models per-command:" line around line 270), insert this new subsection:

````markdown
### Environment variable interpolation

Config YAML values support shell-style `${VAR}` and `${VAR:-default}` substitution. Resolution happens once at load time, before pydantic validates the config. This is the recommended way to inject API keys in containerized deployments (Kubernetes Secrets, OpenShift, Docker Compose), where the YAML is baked into the image and secrets arrive as environment variables.

```yaml
llm:
  api_base: "${LITELLM_GATEWAY_URL:-https://openrouter.ai/api/v1}"
  api_key: "${LITELLM_MASTER_KEY}"

embeddings:
  api_base: "${LITELLM_GATEWAY_URL:-https://openrouter.ai/api/v1}"
  api_key: "${LITELLM_MASTER_KEY}"
```

Rules:

- `${VAR}` — fails at startup if `VAR` is unset or empty. Use this for required secrets so missing config breaks loudly instead of sending empty auth headers.
- `${VAR:-default}` — shell `:-` semantics: both unset and empty env vars fall back to `default`. Use for optional fields like `api_base` with a sensible production default.
- `${VAR:-}` — explicit opt-in for empty/unset; yields the empty string.
- `$$` — escape a literal `$`. For example, `$${FOO}` renders as the literal string `${FOO}`.
- Nested tokens (`${FOO_${BAR}}`) are not supported; compose in the shell before starting the process.
````

- [ ] **Step 2: Add interpolation note to CLAUDE.md**

Open `CLAUDE.md`. Find the "References" section near the bottom. The existing bullet for `Config` reads:

```markdown
- Config: `~/.triz-ai/config.yaml` (default), overridable via `--config` CLI flag or `TRIZ_AI_CONFIG` env var
```

Replace it with:

```markdown
- Config: `~/.triz-ai/config.yaml` (default), overridable via `--config` CLI flag or `TRIZ_AI_CONFIG` env var. Values support `${VAR}` and `${VAR:-default}` env var interpolation; `$$` escapes a literal `$`. Unset/empty `${VAR}` (no default) fails at startup with a field-path error (e.g. `llm.api_key: environment variable LITELLM_MASTER_KEY is not set`).
```

- [ ] **Step 3: Verify no markdown lint issues**

Run: `uv run pre-commit run --all-files`

Expected: PASS. (The repo's pre-commit hooks include trim-trailing-whitespace, end-of-files, and other cosmetic checks — they may reformat the new text. Stage any auto-fixes.)

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document env var interpolation in README & CLAUDE.md (#9)"
```

---

## Post-Implementation Checklist

After all six tasks complete, run this final verification before opening the PR:

- [ ] `uv run pytest` — full test suite passes
- [ ] `uvx ty check src/` — no new type errors
- [ ] `uv run pre-commit run --all-files` — lint/format clean
- [ ] Manual smoke test:
  - Create `/tmp/triz-test.yaml` with `llm:\n  api_key: "${TRIZ_SMOKE:-ok}"\n`
  - Run: `TRIZ_AI_CONFIG=/tmp/triz-test.yaml uv run python -c "from triz_ai.config import load_config; print(load_config().llm.api_key)"`
  - Expected output: `ok`
  - Set `TRIZ_SMOKE=resolved` in env, repeat — expected: `resolved`
  - Replace value with `"${TRIZ_MISSING}"`, unset the var, repeat — expected: `ConfigError` mentioning `llm.api_key` and `TRIZ_MISSING`
