"""
Tests for _call_with_retry — rate limits, unsupported tools, empty choices.

Behaviours covered:
  - Unsupported tools: not retried, raises RuntimeError immediately
  - Rate limits: retried with exponential backoff starting at 65s
  - Empty choices (API content filter refusal): backend.append_user() nudge
    sent once and retried; if still empty choices, raises with clear message
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from schematica.agent import _call_with_retry


_UNSUPPORTED_TOOLS_MSG = (
    "some-provider does not support parameters: ['tools'], "
    "for model=some-vendor/some-model. "
    "To drop these, set `litellm.drop_params=True`"
)

_RATE_LIMIT_MSG = "Rate limit exceeded (429)"
_EMPTY_CHOICES_MSG = "LiteLLM returned empty choices"


class _MockResponse:
    """Minimal stand-in for a successful LLM response."""


class _CountingBackend:
    """Backend that raises a given exception every time call() is invoked."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.call_count = 0

    def call(self, tools, max_tokens):
        self.call_count += 1
        raise self.exc

    def append_user(self, msg: str) -> None:
        pass


class _NudgeableBackend:
    """Backend that raises on the first N calls, then returns a mock response.

    Tracks append_user calls so nudge behaviour can be asserted.
    """

    def __init__(self, raise_on_first: int, exc: Exception) -> None:
        self._raise_on_first = raise_on_first
        self._exc = exc
        self.call_count = 0
        self.nudge_messages: list[str] = []

    def call(self, tools, max_tokens):
        self.call_count += 1
        if self.call_count <= self._raise_on_first:
            raise self._exc
        return _MockResponse()

    def append_user(self, msg: str) -> None:
        self.nudge_messages.append(msg)


# ── unsupported tools — not retried ───────────────────────────────────────────

def test_unsupported_tools_error_calls_backend_exactly_once():
    backend = _CountingBackend(Exception(_UNSUPPORTED_TOOLS_MSG))

    with pytest.raises(RuntimeError):
        _call_with_retry(backend, [])

    assert backend.call_count == 1


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


# ── rate limits — retried with backoff ────────────────────────────────────────

def test_rate_limit_error_retries_more_than_once():
    backend = _CountingBackend(Exception(_RATE_LIMIT_MSG))

    with patch("time.sleep"), patch("schematica.agent.console"):
        with pytest.raises(Exception):
            _call_with_retry(backend, [], max_attempts=3)

    assert backend.call_count > 1


def test_rate_limit_initial_backoff_is_65_seconds():
    # Starting own-backoff must be >= 60s to clear a 60-second rate-limit window
    # on the first retry rather than racing the reset.
    backend = _NudgeableBackend(raise_on_first=1, exc=Exception(_RATE_LIMIT_MSG))

    with patch("time.sleep") as mock_sleep, patch("schematica.agent.console"):
        _call_with_retry(backend, [])

    first_sleep = mock_sleep.call_args_list[0][0][0]
    assert first_sleep >= 60


# ── empty choices — nudge once, then give up ──────────────────────────────────

def test_empty_choices_triggers_append_user_nudge():
    # A single empty-choices response must cause append_user to be called once.
    backend = _NudgeableBackend(raise_on_first=1, exc=Exception(_EMPTY_CHOICES_MSG))

    with patch("schematica.agent.console"):
        _call_with_retry(backend, [])

    assert len(backend.nudge_messages) == 1


def test_empty_choices_nudge_message_is_nonempty():
    backend = _NudgeableBackend(raise_on_first=1, exc=Exception(_EMPTY_CHOICES_MSG))

    with patch("schematica.agent.console"):
        _call_with_retry(backend, [])

    assert backend.nudge_messages[0].strip()


def test_empty_choices_retries_call_after_nudge():
    # call() should be invoked twice: once (empty), once after the nudge (success).
    backend = _NudgeableBackend(raise_on_first=1, exc=Exception(_EMPTY_CHOICES_MSG))

    with patch("schematica.agent.console"):
        _call_with_retry(backend, [])

    assert backend.call_count == 2


def test_empty_choices_returns_result_after_successful_nudge():
    backend = _NudgeableBackend(raise_on_first=1, exc=Exception(_EMPTY_CHOICES_MSG))

    with patch("schematica.agent.console"):
        result = _call_with_retry(backend, [])

    assert isinstance(result, _MockResponse)


def test_empty_choices_twice_raises():
    # Two consecutive empty-choices responses must raise — nudging didn't help.
    backend = _CountingBackend(Exception(_EMPTY_CHOICES_MSG))

    with patch("schematica.agent.console"):
        with pytest.raises(Exception, match="empty"):
            _call_with_retry(backend, [])


def test_empty_choices_twice_calls_backend_exactly_twice():
    backend = _CountingBackend(Exception(_EMPTY_CHOICES_MSG))

    with patch("schematica.agent.console"):
        with pytest.raises(Exception):
            _call_with_retry(backend, [])

    assert backend.call_count == 2


def test_empty_choices_does_not_sleep():
    # Empty-choices is not a rate limit — no sleep between nudge attempts.
    backend = _CountingBackend(Exception(_EMPTY_CHOICES_MSG))

    with patch("time.sleep") as mock_sleep, patch("schematica.agent.console"):
        with pytest.raises(Exception):
            _call_with_retry(backend, [])

    mock_sleep.assert_not_called()
