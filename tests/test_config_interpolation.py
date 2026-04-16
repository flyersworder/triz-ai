"""Tests for YAML config environment variable interpolation."""

import pytest

from triz_ai.config import ConfigError, _interpolate_env, _resolve_tokens


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


def test_zero_string_value_is_truthy_not_empty(monkeypatch):
    # "0" is a truthy non-empty string — must be used as the env value,
    # NOT treated as empty/unset (which would trigger the default or raise).
    monkeypatch.setenv("TRIZ_ZERO", "0")
    assert _resolve_tokens("${TRIZ_ZERO}", "f") == "0"
    assert _resolve_tokens("${TRIZ_ZERO:-fallback}", "f") == "0"


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


# ---------------------------------------------------------------------------
# _interpolate_env walker tests
# ---------------------------------------------------------------------------


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


def test_walk_root_string(monkeypatch):
    # Docstring advertises root-level string handling via <root> sentinel
    monkeypatch.setenv("TRIZ_KEY", "val")
    assert _interpolate_env("${TRIZ_KEY}") == "val"


def test_walk_root_string_error_uses_root_sentinel(monkeypatch):
    # Root-level bare string with unset var: error path should name <root>
    monkeypatch.delenv("TRIZ_MISSING_ROOT", raising=False)
    with pytest.raises(ConfigError, match=r"<root>.*TRIZ_MISSING_ROOT"):
        _interpolate_env("${TRIZ_MISSING_ROOT}")
