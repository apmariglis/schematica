"""
Tests for _format_iter_stats — per-iteration token/cost/throughput display.

The output is a box with two rows and an explanatory footer:
  - "current iter N/M" header row: per-iteration tokens, cost, duration
  - "accumulated"      row: session-total tokens, cost, elapsed time, iter/min
  - footer: note that each iteration is a single LLM call

All data is known only after the call returns, so the numbers are always accurate.
"""
from __future__ import annotations

import pytest

from schematica.agent import _format_iter_stats, _RequestTracker


_MOCK_PRICING = {
    "test-model": {
        "input":  1.0,   # $1.00 per 1M input tokens
        "output": 2.0,   # $2.00 per 1M output tokens
    }
}


def _tracker_with(n_requests: int, started_at: float = 0.0, now: float = 60.0) -> _RequestTracker:
    t = _RequestTracker(started_at)
    for i in range(n_requests):
        t.record(now=now)
    return t


# ── token counts ──────────────────────────────────────────────────────────────

def test_input_token_count_appears_with_comma_separator():
    result = _format_iter_stats(1500, 200, "test-model", _MOCK_PRICING)
    assert "1,500" in result


def test_output_token_count_appears_with_comma_separator():
    result = _format_iter_stats(1000, 2500, "test-model", _MOCK_PRICING)
    assert "2,500" in result


def test_in_and_out_labels_appear():
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    assert "in" in result
    assert "out" in result


# ── cost ──────────────────────────────────────────────────────────────────────

def test_cost_appears_in_output():
    result = _format_iter_stats(1000, 500, "test-model", _MOCK_PRICING)
    assert "$" in result


def test_zero_tokens_does_not_crash():
    result = _format_iter_stats(0, 0, "test-model", _MOCK_PRICING)
    assert "0" in result


# ── duration ──────────────────────────────────────────────────────────────────

def test_iter_duration_under_60s_shown_as_seconds():
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING, iter_duration=7.3)
    assert "7.3s" in result


def test_iter_duration_over_60s_shown_as_minutes_and_seconds():
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING, iter_duration=65.0)
    assert "1m05s" in result


def test_iter_duration_exactly_60s_shown_as_minutes():
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING, iter_duration=60.0)
    assert "1m00s" in result


# ── box structure ─────────────────────────────────────────────────────────────

def test_output_is_a_string():
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    assert isinstance(result, str)


def test_box_top_left_corner_present():
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    assert "╭" in result


def test_box_bottom_left_corner_present():
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    assert "╰" in result


def test_box_current_iter_label_present():
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    assert "current iter" in result


def test_box_header_shows_iteration_number_when_provided():
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING, iter_num=3, max_iter=31)
    assert "current iter 3/31" in result


def test_box_header_omits_iteration_number_when_not_provided():
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    assert "current iter" in result
    assert "/" not in result.splitlines()[0]


def test_box_is_multiline():
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    assert "\n" in result


def test_separator_inside_box_is_middle_dot():
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    assert "·" in result


def test_footer_explains_single_llm_call_per_iteration():
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    assert "1 LLM call" in result


# ── accumulated row (requires tracker) ────────────────────────────────────────

def test_accumulated_row_absent_without_tracker():
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    assert "accumulated" not in result


def test_accumulated_row_present_with_tracker():
    tracker = _tracker_with(1, started_at=0.0, now=10.0)
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING, tracker=tracker, now=10.0)
    assert "accumulated" in result


def test_total_cost_appears_in_accumulated_row():
    tracker = _tracker_with(1, started_at=0.0, now=10.0)
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        tracker=tracker, now=10.0,
        total_cost=0.0123,
    )
    assert "0.0123" in result


def test_accumulated_token_counts_appear_in_accumulated_row():
    tracker = _tracker_with(1, started_at=0.0, now=10.0)
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        tracker=tracker, now=10.0,
        total_in=5000, total_out=1200,
    )
    assert "5,000" in result
    assert "1,200" in result


def test_elapsed_time_appears_in_accumulated_row():
    # tracker started at 0, now=45 → elapsed = 45s
    tracker = _tracker_with(1, started_at=0.0, now=45.0)
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING, tracker=tracker, now=45.0)
    assert "45.0s" in result


def test_iter_per_min_appears_in_accumulated_row():
    # 3 iterations over 60s → 3.0 iter/min
    tracker = _tracker_with(3, started_at=0.0, now=60.0)
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        tracker=tracker, now=60.0,
    )
    lines = result.splitlines()
    accumulated_line = lines[3]    # fourth line is the accumulated content row
    assert "3.0 iter/min" in accumulated_line


def test_no_tracker_still_returns_string():
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    assert isinstance(result, str)


# ── context window fill % ─────────────────────────────────────────────────────

def test_context_fill_pct_appears_when_context_window_provided():
    # 20,000 tokens sent into a 200,000 token window → 10.0%
    result = _format_iter_stats(
        20_000, 500, "test-model", _MOCK_PRICING,
        context_window=200_000,
    )
    assert "10.0%" in result


def test_context_fill_pct_absent_when_context_window_not_provided():
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    assert "%" not in result


def test_context_fill_pct_is_in_current_iter_row():
    result = _format_iter_stats(
        10_000, 200, "test-model", _MOCK_PRICING,
        context_window=100_000,
    )
    current_iter_line = result.splitlines()[1]
    assert "10.0%" in current_iter_line


def test_context_fill_pct_rounds_to_one_decimal():
    # 1 token in 3 token window → 33.333...% → shown as 33.3%
    result = _format_iter_stats(
        1, 0, "test-model", _MOCK_PRICING,
        context_window=3,
    )
    assert "33.3%" in result


# ── removed fields ────────────────────────────────────────────────────────────

def test_request_count_not_shown_separately():
    # Redundant with iteration number in header — removed
    tracker = _tracker_with(4, started_at=0.0, now=40.0)
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING, tracker=tracker, now=40.0)
    assert "requests" not in result


def test_instantaneous_rpm_not_shown_in_current_iter_row():
    # Instantaneous rpm removed — only iter/min average in accumulated
    tracker = _tracker_with(1, started_at=0.0, now=6.0)
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        tracker=tracker, now=6.0,
        iter_duration=6.0,
    )
    current_iter_line = result.splitlines()[1]
    assert "rpm" not in current_iter_line


def test_in_minute_window_count_not_shown():
    # Per-minute window counter removed — replaced by iter/min average
    tracker = _tracker_with(3, started_at=0.0, now=30.0)
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING, tracker=tracker, now=30.0)
    assert "/min" not in result or "iter/min" in result   # only iter/min is allowed


# ── all lines in box have equal width ─────────────────────────────────────────

def test_all_box_lines_have_equal_width():
    tracker = _tracker_with(3, started_at=0.0, now=90.0)
    result = _format_iter_stats(
        1500, 800, "test-model", _MOCK_PRICING,
        tracker=tracker, now=90.0,
        iter_duration=12.5,
        total_in=4500, total_out=2400,
        total_cost=0.0089,
    )
    lines = result.splitlines()
    assert len(set(len(line) for line in lines)) == 1, (
        f"Box lines have unequal widths: {[len(l) for l in lines]}"
    )
