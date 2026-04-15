"""
Tests for _format_iter_stats with prompt-cache token accounting.

When SC_CACHE=true (Anthropic native backend), prompt caching means almost all
input tokens come back as cache_read_tokens (charged at 10% of input price),
not as input_tokens. Without accounting for this:

  - "2 in" appears instead of "51,234 in" — confusing display
  - cost is underreported (cache_read tokens are still charged, just cheaper)
  - context fill % shows ~0% instead of the actual ~25%

The fix threads cache_creation_tokens and cache_read_tokens through
_format_iter_stats and the cost calc in _run_phase.
"""
from __future__ import annotations

from schematica.agent import _format_iter_stats, _RequestTracker


_MOCK_PRICING = {
    "test-model": {
        "input":       1.00,   # $1.00 / 1M input tokens
        "output":      2.00,   # $2.00 / 1M output tokens
        "cache_write": 1.25,   # $1.25 / 1M cache-creation tokens
        "cache_read":  0.10,   # $0.10 / 1M cache-read tokens
    }
}


# ── cost calculation includes cache tokens ─────────────────────────────────────

def test_cost_includes_cache_read_tokens():
    # 0 fresh in, 0 out, 1,000,000 cache-read → $0.10
    result = _format_iter_stats(
        0, 0, "test-model", _MOCK_PRICING,
        cache_read_tokens=1_000_000,
    )
    assert "0.1000" in result


def test_cost_includes_cache_creation_tokens():
    # 0 fresh in, 0 out, 1,000,000 cache-write → $1.25
    result = _format_iter_stats(
        0, 0, "test-model", _MOCK_PRICING,
        cache_creation_tokens=1_000_000,
    )
    assert "1.2500" in result


def test_cost_sums_all_token_types():
    # 100k fresh in ($0.10) + 100k cache_write ($0.125) + 100k cache_read ($0.01) + 50k out ($0.10) = $0.335
    result = _format_iter_stats(
        100_000, 50_000, "test-model", _MOCK_PRICING,
        cache_creation_tokens=100_000,
        cache_read_tokens=100_000,
    )
    assert "0.3350" in result


def test_zero_cache_tokens_does_not_affect_cost():
    # Same as no cache-token args: 1000 in + 500 out = $0.001 + $0.001 = $0.002
    result_no_args = _format_iter_stats(1000, 500, "test-model", _MOCK_PRICING)
    result_zero    = _format_iter_stats(
        1000, 500, "test-model", _MOCK_PRICING,
        cache_creation_tokens=0, cache_read_tokens=0,
    )
    # extract the $ figure from each
    import re
    cost_no_args = re.search(r"\$[\d.]+", result_no_args).group()
    cost_zero    = re.search(r"\$[\d.]+", result_zero).group()
    assert cost_no_args == cost_zero


# ── display shows cached token count ──────────────────────────────────────────

def test_cache_read_tokens_shown_when_nonzero():
    result = _format_iter_stats(
        2, 710, "test-model", _MOCK_PRICING,
        cache_read_tokens=51_234,
    )
    assert "51,234" in result
    assert "cached" in result


def test_cache_creation_tokens_shown_when_nonzero():
    result = _format_iter_stats(
        50_000, 200, "test-model", _MOCK_PRICING,
        cache_creation_tokens=50_000,
    )
    assert "50,000" in result
    assert "cache" in result


def test_cached_label_absent_when_no_cache_tokens():
    result = _format_iter_stats(1000, 500, "test-model", _MOCK_PRICING)
    assert "cached" not in result


def test_cache_write_label_absent_when_no_cache_creation_tokens():
    result = _format_iter_stats(1000, 500, "test-model", _MOCK_PRICING)
    assert "cache↑" not in result


# ── context fill % counts total effective input ────────────────────────────────

def test_context_fill_includes_cache_read_tokens():
    # 2 fresh + 49,998 cached = 50,000 effective input in 200,000 window → 25.0%
    result = _format_iter_stats(
        2, 100, "test-model", _MOCK_PRICING,
        cache_read_tokens=49_998,
        context_window=200_000,
    )
    assert "25.0%" in result


def test_context_fill_includes_cache_creation_tokens():
    # 1,000 fresh + 49,000 cache_write = 50,000 effective in 200,000 window → 25.0%
    result = _format_iter_stats(
        1_000, 100, "test-model", _MOCK_PRICING,
        cache_creation_tokens=49_000,
        context_window=200_000,
    )
    assert "25.0%" in result


def test_context_fill_without_cache_unchanged():
    # Existing behaviour: 20,000 fresh in a 200,000 window → 10.0%
    result = _format_iter_stats(
        20_000, 500, "test-model", _MOCK_PRICING,
        context_window=200_000,
    )
    assert "10.0%" in result


# ── box lines still have equal width with cache tokens ─────────────────────────

def test_all_box_lines_equal_width_with_cache_tokens():
    tracker = _RequestTracker(started_at=0.0)
    tracker.record(now=10.0)
    result = _format_iter_stats(
        2, 710, "test-model", _MOCK_PRICING,
        cache_read_tokens=51_234,
        tracker=tracker, now=10.0,
        iter_duration=5.3,
        total_in=2, total_out=710,
        total_cost=0.0036,
        iter_num=1, max_iter=37,
        context_window=200_000,
    )
    lines = result.splitlines()
    assert len(set(len(line) for line in lines)) == 1, (
        f"Box lines have unequal widths: {[len(l) for l in lines]}"
    )
