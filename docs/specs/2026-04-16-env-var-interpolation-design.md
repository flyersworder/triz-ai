# Config Environment Variable Interpolation Design

**Date**: 2026-04-16
**Status**: Draft
**Issue**: [#9](https://github.com/flyersworder/triz-ai/issues/9)

## Overview

Support shell-style environment variable interpolation (`${VAR}` and `${VAR:-default}`) inside values of the YAML config file. Resolution runs once in `load_config()`, between `yaml.safe_load()` and `Settings(**data)`, as a pure preprocessing step. The pydantic models, `LLMClient`, CLI, and existing tests are unchanged.

This unblocks K8s / OpenShift deployments where API keys and other secrets are injected as environment variables from `Secret` resources and must not be hardcoded in the config YAML or baked into Docker images.

### Goals

- **Deployability**: config YAML can reference env-injected secrets without hardcoding them.
- **Standard syntax**: use Docker Compose / Helm convention (`${VAR}`, `${VAR:-default}`, `$$` escape) so DevOps engineers recognize it immediately.
- **Fail fast**: an unset `${VAR}` (with no default) raises a clear error at startup, naming the exact config field — never silently produces empty strings that get sent as auth headers.
- **No new dependencies**: implementation is ~40 lines of hand-written scanning logic. Alternative libraries were surveyed; none cleanly fit the chosen grammar or integration point.
- **Backward compatible**: configs without `${...}` tokens pass through unchanged.

### Non-Goals

- Replacing or extending pydantic-settings. This is a preprocessing layer on YAML, not a new settings source.
- Nested interpolation (`${FOO_${BAR}}`). Users who need composition pre-export a composed env var.
- Interpolation of dict keys. Keys are schema, not data.
- Interpolation of non-string scalars (ints, bools, None). YAML has already typed them.
- A plugin API for custom transformations (`${VAR,,}`, `${VAR//a/b}`, etc.). Can be added later if needed.

## Architecture

### Placement: preprocessing in `load_config()`

```
load_config(path)
  ├─ yaml.safe_load(file)              → raw dict with ${...} tokens
  ├─ _interpolate_env(raw_dict)        → dict with tokens resolved  ← NEW
  └─ Settings(**resolved_dict)         → validated pydantic model
```

Nothing downstream (LLMClient, CLI, tests, pydantic models) knows interpolation exists. `--config`, `TRIZ_AI_CONFIG`, and programmatic `load_config(path)` callers all get resolution for free.

### Why not pydantic field validators

Scattering resolution across field validators couples validation concerns (type coercion, range checks) with resolution concerns (env var lookup). Each new config field would need to remember the validator. A single preprocessing pass keeps the schema declarative.

### Why not a library

Surveyed (2026-04-16):

| Option | Syntax | Integration | Maturity | Verdict |
|---|---|---|---|---|
| `pydantic-settings-extra-sources` | Exact `${VAR:-default}` | Native | **0 stars, 1 maintainer, 2025** | Rejected: supply-chain risk in an auth-adjacent path |
| `pyaml-env` | `${VAR:default}`, requires `!ENV` tag | None | Maintained | Rejected: `!ENV` tag on every value is verbose and silent-fails if forgotten; default sentinel is literal `"N/A"` |
| `OmegaConf` | `${oc.env:VAR,default}` | None (replaces pydantic) | Mature (Meta) | Rejected: different syntax, drags in Hydra-style config framework |
| `dynaconf` | `@format` / `@jinja` | Replaces pydantic-settings | Mature | Rejected: philosophy mismatch |
| `Dependency Injector` | `${VAR:default}` | DI framework | Mature | Rejected: overkill (full DI) |
| Hand-written scanner | Exact (we define) | Drop-in preprocessing | ~40 LOC we own | **Chosen** |

The scanner trades ~40 lines of local code for: exact control of syntax, field-path error messages (`"Config field llm.api_key: env var LITELLM_MASTER_KEY is not set"`), zero dependency surface, and no risk of upstream abandonment.

## Grammar

Implemented as a character-by-character scanner (not regex) so the grammar can grow without rewrite.

### Tokens

| Input | Output | Notes |
|---|---|---|
| `$$` | `$` | Escape. `$${VAR}` yields literal `${VAR}`. |
| `${VAR}` | value of `os.environ["VAR"]` | Raises if `VAR` is unset OR empty. Matches the fail-fast safety goal: if the caller wants to tolerate empty/unset, they write `${VAR:-}`. This diverges from POSIX shell (where `${VAR}` is permissive) in favor of catching missing-secret deployment bugs early. |
| `${VAR:-default}` | `os.environ.get("VAR") or default` | Shell `:-` semantics: both unset AND empty-string env vars fall back to `default`. `default` is a literal; not re-interpolated. |
| `${VAR:-}` | `""` if `VAR` unset or empty, else its value | Explicit empty default; useful for optional fields. |

`VAR` matches `[A-Za-z_][A-Za-z0-9_]*`.

### Scope

- Applied to every **string leaf** in the parsed dict, walked recursively.
- **List elements** are walked (path notation `key[i]`).
- **Dict keys** are NOT interpolated.
- **Non-string scalars** (int, bool, None, float) pass through untouched.
- **Multiple tokens per string** are supported: `"https://${HOST}:${PORT:-443}/v1"`.

### Non-features

- Nested tokens (`${FOO_${BAR}}`) are not supported. The scanner matches the first `}` after `${`, so `${FOO_${BAR}}` parses as `${FOO_${BAR}` + trailing `}`, which raises on the odd `${` inside. Users compose in the shell.
- Default values cannot contain `}` or `${`. `default` is read verbatim from after `:-` up to the first `}`.
- No transformation operators (uppercase, substring, etc.). Can be added later by extending the scanner.

## Implementation

### New file-local helpers in `src/triz_ai/config.py`

```python
class ConfigError(Exception):
    """Raised when config loading or interpolation fails."""

def _interpolate(value: str, field_path: str) -> str:
    """Resolve ${VAR} and ${VAR:-default} tokens in `value`.

    `field_path` is used for error messages (e.g. "llm.api_key").
    Raises ConfigError on unclosed tokens, empty names, or unset vars
    without defaults.
    """
    # char-by-char scanner: handle $$, ${VAR}, ${VAR:-default}

def _walk(data, field_path: str = ""):
    """Recursively walk dicts/lists, interpolating string leaves."""
    # dict → recurse on values with "{path}.{key}"
    # list → recurse on items with "{path}[{i}]"
    # str  → return _interpolate(data, field_path)
    # else → return unchanged
```

### Call site in `load_config()`

```python
if config_path.exists():
    import yaml
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    data = _walk(data)           # ← single new line
    return Settings(**data)
return Settings()
```

### Error types

- **Unclosed `${`**: `ConfigError("Config field {path}: unclosed '${' in value {value!r}")`
- **Empty name** (`${}`, `${:-x}`): `ConfigError("Config field {path}: empty variable name in {value!r}")`
- **Unset or empty var, no default**: `ConfigError("Config field {path}: environment variable {VAR} is not set (or is empty). Either set it to a non-empty value, or provide a default: ${{{VAR}:-}}")`

`ConfigError` is raised from `load_config()` — before `LLMClient` instantiates — so no circular-import concern. CLI top-level error handling already catches broad exceptions and prints; no extra wiring needed.

## Testing

New file: `tests/test_config_interpolation.py`.

### Test groups

1. **Happy path**
   - Single `${VAR}` resolves from env
   - Multiple tokens in one string: `"http://${HOST}:${PORT}"`
   - Nested dicts and lists walked correctly
   - Non-string scalars (int, bool, None) pass through unchanged

2. **Defaults**
   - `${VAR:-default}` uses default when VAR unset
   - `${VAR:-default}` uses default when VAR is empty string (shell `:-` semantics)
   - `${VAR:-}` yields empty string when VAR unset or empty
   - `${VAR:-default}` ignored when VAR is set to a non-empty value (uses env value)
   - `${VAR}` with VAR set to empty string raises `ConfigError` (fail-fast safety; use `${VAR:-}` to allow empty)

3. **Escape**
   - `$$` → `$`
   - `$${FOO}` → literal `${FOO}` (no interpolation)
   - `$$$${FOO}` → literal `$${FOO}`
   - `${FOO}$${BAR}` with FOO=x → `x${BAR}`

4. **Errors**
   - Unset `${VAR}` without default → `ConfigError` mentioning field path and var name
   - Empty-string `${VAR}` without default → same `ConfigError` (fail-fast safety)
   - Unclosed `${FOO` → `ConfigError`
   - Empty name `${}` and `${:-x}` → `ConfigError`
   - Error message for nested field includes full dotted path (e.g. `embeddings.api_base`)
   - Error message for list element includes index (e.g. `retries[2]`)

5. **Integration**
   - Write temp YAML with `${...}` tokens, set env vars via `monkeypatch.setenv`, call `load_config(path)`, assert resolved `Settings` values on `llm.api_key`, `llm.api_base`, etc.
   - Backward compat: existing YAML without tokens loads identically to pre-change behavior.

Uses `pytest`'s `monkeypatch.setenv` / `monkeypatch.delenv` and `tmp_path`. No new test dependencies.

## Documentation

1. `CLAUDE.md` — add one line under "References":
   > Config YAML values support `${VAR}` and `${VAR:-default}` env var interpolation; `$$` escapes a literal `$`.

2. README — if it documents config (to be verified during implementation), add a one-paragraph K8s/OpenShift example:
   ```yaml
   llm:
     api_base: "${LITELLM_GATEWAY_URL:-https://openrouter.ai/api/v1}"
     api_key: "${LITELLM_MASTER_KEY}"
   ```

## Rollout

- Purely additive; no migration.
- Existing YAML files without `${...}` tokens are unaffected.
- No CLI changes, no config schema changes, no dependency changes.
- Version bump: patch-level (bug-fix / feature addition) — follow existing release conventions.

## Open questions

None at spec time. Any gray-area behavior (nested tokens, defaults-with-braces) is explicitly called out as non-goal in the Grammar section.
