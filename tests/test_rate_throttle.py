"""
Tests for output-token rate-limit throttling:

  _retry_after_seconds   — extracts the retry-after hint from a 429 exception
  _OutputTokenBucket     — rolling-window tracker with proactive wait logic
  _call_with_retry       — uses the retry-after hint instead of fixed backoff

These helpers prevent the multi-hour stalls caused by generating large
finish_catalogue JSON payloads against models with low output-TPM limits.
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from schematica.agent import _OutputTokenBucket, _retry_after_seconds


# ── _retry_after_seconds ──────────────────────────────────────────────────────

def _exc_with_header(key: str, value: str) -> Exception:
    """Build a mock exception whose .response.headers contains one entry."""
    exc = Exception("rate limit")
    exc.response = MagicMock()
    exc.response.headers = {key: value}
    return exc


def test_retry_after_returns_none_when_exception_has_no_response():
    exc = Exception("plain error")

    assert _retry_after_seconds(exc) is None


def test_retry_after_returns_none_when_response_has_no_headers():
    exc = Exception("rate limit")
    exc.response = MagicMock()
    exc.response.headers = {}

    assert _retry_after_seconds(exc) is None


def test_retry_after_returns_float_from_retry_after_header():
    exc = _exc_with_header("retry-after", "45")

    assert _retry_after_seconds(exc) == 45.0


def test_retry_after_returns_float_from_fractional_value():
    exc = _exc_with_header("retry-after", "12.5")

    assert _retry_after_seconds(exc) == 12.5


def test_retry_after_returns_none_when_header_is_non_numeric():
    exc = _exc_with_header("retry-after", "soon")

    assert _retry_after_seconds(exc) is None


def test_retry_after_returns_none_when_response_attribute_is_none():
    exc = Exception("rate limit")
    exc.response = None

    assert _retry_after_seconds(exc) is None


# ── _OutputTokenBucket — recording and window ─────────────────────────────────

def test_bucket_starts_with_zero_tokens_in_window():
    bucket = _OutputTokenBucket(limit=16_000)

    assert bucket.tokens_in_window(now=0.0) == 0


def test_bucket_records_tokens_and_reports_them_in_window():
    bucket = _OutputTokenBucket(limit=16_000)
    bucket.record(now=0.0, tokens=500)

    assert bucket.tokens_in_window(now=0.0) == 500


def test_bucket_accumulates_multiple_records():
    bucket = _OutputTokenBucket(limit=16_000)
    bucket.record(now=0.0, tokens=300)
    bucket.record(now=5.0, tokens=700)

    assert bucket.tokens_in_window(now=5.0) == 1000


def test_bucket_evicts_entries_older_than_60_seconds():
    bucket = _OutputTokenBucket(limit=16_000)
    bucket.record(now=0.0, tokens=500)
    bucket.record(now=30.0, tokens=200)

    # At t=65, the t=0 entry is 65s old — evicted; t=30 entry is 35s old — kept
    assert bucket.tokens_in_window(now=65.0) == 200


def test_bucket_evicts_all_entries_after_full_window():
    bucket = _OutputTokenBucket(limit=16_000)
    bucket.record(now=0.0, tokens=500)

    # At t=61, entry is 61s old — past the 60s window
    assert bucket.tokens_in_window(now=61.0) == 0


def test_bucket_ignores_zero_token_records():
    bucket = _OutputTokenBucket(limit=16_000)
    bucket.record(now=0.0, tokens=0)

    assert bucket.tokens_in_window(now=0.0) == 0


# ── _OutputTokenBucket — limit update ────────────────────────────────────────

def test_bucket_update_limit_changes_the_limit():
    bucket = _OutputTokenBucket(limit=16_000)
    bucket.update_limit(40_000)

    # With 30k tokens recorded and a 40k limit there is headroom
    bucket.record(now=0.0, tokens=30_000)
    with patch("time.sleep") as mock_sleep:
        waited = bucket.proactive_wait(now=0.0, expected=5_000)
    assert mock_sleep.call_count == 0
    assert waited == 0.0


# ── _OutputTokenBucket — proactive_wait ──────────────────────────────────────

def test_proactive_wait_returns_zero_when_budget_is_available():
    bucket = _OutputTokenBucket(limit=16_000)
    bucket.record(now=0.0, tokens=5_000)

    with patch("time.sleep") as mock_sleep:
        # 5k used + 8k expected = 13k, well under 16k limit
        waited = bucket.proactive_wait(now=0.0, expected=8_000)

    assert mock_sleep.call_count == 0
    assert waited == 0.0


def test_proactive_wait_returns_zero_when_window_is_empty():
    bucket = _OutputTokenBucket(limit=16_000)

    with patch("time.sleep") as mock_sleep:
        waited = bucket.proactive_wait(now=0.0, expected=10_000)

    assert mock_sleep.call_count == 0
    assert waited == 0.0


def test_proactive_wait_sleeps_when_used_plus_expected_exceeds_limit():
    # 14k already used, 5k expected → 19k > 16k limit — must wait
    bucket = _OutputTokenBucket(limit=16_000)
    bucket.record(now=0.0, tokens=14_000)

    with patch("time.sleep") as mock_sleep:
        waited = bucket.proactive_wait(now=0.0, expected=5_000)

    assert mock_sleep.call_count == 1


def test_proactive_wait_returns_positive_seconds_when_sleep_was_needed():
    bucket = _OutputTokenBucket(limit=16_000)
    bucket.record(now=0.0, tokens=14_000)

    with patch("time.sleep"):
        waited = bucket.proactive_wait(now=0.0, expected=5_000)

    assert waited > 0.0


def test_proactive_wait_passes_correct_duration_to_sleep():
    # Token recorded at t=0, window=60s; at t=10 it has 50s of life left.
    # We need it to evict to have headroom → wait = 60 - 10 + 0.5 = 50.5s
    bucket = _OutputTokenBucket(limit=16_000)
    bucket.record(now=0.0, tokens=14_000)

    with patch("time.sleep") as mock_sleep:
        bucket.proactive_wait(now=10.0, expected=5_000)

    expected_wait = pytest.approx(50.5, abs=0.1)
    actual_wait = mock_sleep.call_args[0][0]
    assert actual_wait == expected_wait


def test_proactive_wait_only_waits_for_entries_needed_to_free_headroom():
    # Two entries: 8k at t=0 and 8k at t=30.
    # Expected: 5k. Used: 16k. Need to drop 5k.
    # Oldest entry (8k at t=0) is enough to cover the 5k shortfall.
    # At t=10: oldest entry has 50s of life left → wait = 50 + 0.5 = 50.5s.
    # We should NOT need to wait for the t=30 entry (which has 80s left at t=10).
    bucket = _OutputTokenBucket(limit=16_000)
    bucket.record(now=0.0, tokens=8_000)
    bucket.record(now=30.0, tokens=8_000)

    with patch("time.sleep") as mock_sleep:
        bucket.proactive_wait(now=10.0, expected=5_000)

    actual_wait = mock_sleep.call_args[0][0]
    # Should wait ~50.5s (for oldest entry), not ~80.5s (for newest entry)
    assert actual_wait == pytest.approx(50.5, abs=0.1)
