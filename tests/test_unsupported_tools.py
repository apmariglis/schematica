"""
Tests for _call_with_retry behaviour when a model doesn't support tool calls.

When LiteLLM raises UnsupportedParamsError (or any error whose message
contains "does not support parameters" and "tools"), schematica should:
  - NOT retry (it's a config error, not transient)
  - raise a clear RuntimeError explaining that the model lacks tool-call support
  - call backend.call() exactly once
"""
from __future__ import annotations

import pytest

from schematica.agent import _call_with_retry


_UNSUPPORTED_TOOLS_MSG = (
    "together_ai does not support parameters: ['tools'], "
    "for model=meta-llama/Llama-3.1-405B-Instruct-Turbo. "
    "To drop these, set `litellm.drop_params=True`"
)

_RATE_LIMIT_MSG = "Rate limit exceeded (429)"


class _CountingBackend:
    """Backend that raises a given exception every time call() is invoked."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.call_count = 0

    def call(self, tools, max_tokens):
        self.call_count += 1
        raise self.exc


# ── does not retry ────────────────────────────────────────────────────────────

def test_unsupported_tools_error_calls_backend_exactly_once():
    backend = _CountingBackend(Exception(_UNSUPPORTED_TOOLS_MSG))

    with pytest.raises(RuntimeError):
        _call_with_retry(backend, [])

    assert backend.call_count == 1


def test_rate_limit_error_retries_more_than_once():
    # Sanity check: rate-limit errors ARE retried (contrast with tools error)
    backend = _CountingBackend(Exception(_RATE_LIMIT_MSG))

    with pytest.raises(Exception):
        _call_with_retry(backend, [], max_attempts=3)

    assert backend.call_count > 1


# ── raises RuntimeError with a clear message ──────────────────────────────────

def test_unsupported_tools_error_raises_runtime_error():
    backend = _CountingBackend(Exception(_UNSUPPORTED_TOOLS_MSG))

    with pytest.raises(RuntimeError):
        _call_with_retry(backend, [])


def test_unsupported_tools_error_message_mentions_tool_calls():
    backend = _CountingBackend(Exception(_UNSUPPORTED_TOOLS_MSG))

    with pytest.raises(RuntimeError, match="tool"):
        _call_with_retry(backend, [])


def test_unsupported_tools_error_message_suggests_different_model():
    backend = _CountingBackend(Exception(_UNSUPPORTED_TOOLS_MSG))

    with pytest.raises(RuntimeError, match="model"):
        _call_with_retry(backend, [])
