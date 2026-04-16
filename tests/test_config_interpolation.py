"""Tests for YAML config environment variable interpolation."""

from triz_ai.config import _resolve_tokens


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
