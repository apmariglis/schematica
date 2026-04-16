"""
Tests for _format_iter_stats and _format_phase_summary.

_format_iter_stats produces a box with up to three sections:
  - current iter:        per-call Tokens, cost, duration, context%
  - averages (phase X):  per-phase averages (shown when phase_n > 0)
  - session:             cross-phase totals + llm calls/min (shown when session_tracker provided)

Top border embeds "1 iter = 1 LLM call" on the right.
Bottom border is plain.
Token format across all sections: "Tokens: X in | Y out"
"""
from __future__ import annotations

import pytest

from schematica.agent import _format_iter_stats, _format_phase_summary, _RequestTracker


_MOCK_PRICING = {
    "test-model": {
        "input":  1.0,   # $1.00 per 1M input tokens
        "output": 2.0,   # $2.00 per 1M output tokens
    }
}


def _session_tracker(n_requests: int, started_at: float = 0.0, now: float = 60.0) -> _RequestTracker:
    t = _RequestTracker(started_at)
    for _ in range(n_requests):
        t.record(now=now)
    return t


# ── token counts ──────────────────────────────────────────────────────────────

def test_input_token_count_appears_with_comma_separator():
    result = _format_iter_stats(1500, 200, "test-model", _MOCK_PRICING)
    assert "1,500" in result


def test_output_token_count_appears_with_comma_separator():
    result = _format_iter_stats(1000, 2500, "test-model", _MOCK_PRICING)
    assert "2,500" in result


def test_token_format_uses_tokens_label_and_pipe_separator():
    # "Tokens: X in | Y out" — not "X in · Y out"
    result = _format_iter_stats(1000, 500, "test-model", _MOCK_PRICING)
    assert "Tokens:" in result
    assert "in |" in result or "in|" in result


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


def test_separator_between_groups_is_middle_dot():
    # "·" is used between token group, cost, duration, etc.
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    assert "·" in result


def test_disclaimer_embedded_in_top_border():
    # "1 iter = 1 LLM call" appears in the top border line, not a footer
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    top_line = result.splitlines()[0]
    assert "1 iter" in top_line or "LLM call" in top_line


def test_bottom_border_is_plain():
    # Bottom border has no text embedded — just dashes
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    bottom_line = result.splitlines()[-1]
    assert bottom_line.startswith("╰") and bottom_line.endswith("╯")
    assert "iter" not in bottom_line and "LLM" not in bottom_line


# ── session row (requires tracker) ────────────────────────────────────────────

def test_session_row_absent_without_tracker():
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    assert "session (accumulated)" not in result


def test_session_row_present_with_tracker():
    tracker = _session_tracker(1, started_at=0.0, now=10.0)
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        session_tracker=tracker, now=10.0,
    )
    assert "session (accumulated)" in result


def test_session_cost_appears_in_session_row():
    tracker = _session_tracker(1, started_at=0.0, now=10.0)
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        session_tracker=tracker, now=10.0,
        session_total_cost=0.0123,
    )
    assert "0.0123" in result


def test_session_token_counts_appear_in_session_row():
    tracker = _session_tracker(1, started_at=0.0, now=10.0)
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        session_tracker=tracker, now=10.0,
        session_total_in=5000, session_total_out=1200,
    )
    assert "5,000" in result
    assert "1,200" in result


def test_elapsed_time_appears_in_session_row():
    # tracker started at 0, now=45 → elapsed = 45s
    tracker = _session_tracker(1, started_at=0.0, now=45.0)
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        session_tracker=tracker, now=45.0,
    )
    assert "45.0s" in result


def test_llm_calls_per_min_appears_in_session_row():
    # 3 calls over 60s → 3.0 llm calls/min
    tracker = _session_tracker(3, started_at=0.0, now=60.0)
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        session_tracker=tracker, now=60.0,
        phase_n=3, phase_elapsed=60.0,
        phase_total_in=300, phase_total_out=150,
    )
    session_line = result.splitlines()[5]  # [0]=top,[1]=iter,[2]=avg hdr,[3]=avg,[4]=session hdr,[5]=session
    assert "3.0 llm calls/min" in session_line


def test_llm_calls_per_min_not_iter_per_min():
    # Label is "llm calls/min", not the old "iter/min"
    tracker = _session_tracker(2, started_at=0.0, now=60.0)
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        session_tracker=tracker, now=60.0,
    )
    assert "iter/min" not in result
    assert "llm calls/min" in result


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


def test_context_fill_uses_word_context():
    result = _format_iter_stats(
        10_000, 200, "test-model", _MOCK_PRICING,
        context_window=100_000,
    )
    assert "context" in result


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


# ── averages row (phase-level, resets per phase) ───────────────────────────────

def test_averages_row_absent_without_phase_data():
    # phase_n=0 (default) → no averages section
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    assert "averages" not in result


def test_averages_row_present_when_phase_n_provided():
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        phase_label="1 (exploration)",
        phase_n=2, phase_elapsed=20.0,
        phase_total_in=200, phase_total_out=100, phase_total_cost=0.0004,
    )
    assert "averages" in result


def test_averages_row_includes_phase_label():
    # "1 (exploration)" is rendered as "phase 1 — exploration"
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        phase_label="1 (exploration)",
        phase_n=1, phase_elapsed=10.0,
        phase_total_in=100, phase_total_out=50, phase_total_cost=0.0002,
    )
    assert "phase 1" in result
    assert "exploration" in result


def test_averages_row_shows_avg_input_tokens():
    # phase_n=2, phase_total_in=1000 → avg 500 in
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        phase_label="1", phase_n=2, phase_elapsed=20.0,
        phase_total_in=1000, phase_total_out=200, phase_total_cost=0.0010,
    )
    avg_line = result.splitlines()[3]  # [0]=top,[1]=iter,[2]=avg hdr,[3]=avg content
    assert "500" in avg_line


def test_averages_row_shows_avg_output_tokens():
    # phase_n=4, phase_total_out=2000 → avg 500 out
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        phase_label="1", phase_n=4, phase_elapsed=40.0,
        phase_total_in=400, phase_total_out=2000, phase_total_cost=0.0008,
    )
    avg_line = result.splitlines()[3]
    assert "500" in avg_line


def test_averages_row_shows_avg_cost():
    # phase_n=2, phase_total_cost=0.0100 → avg $0.0050
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        phase_label="1", phase_n=2, phase_elapsed=20.0,
        phase_total_in=200, phase_total_out=100, phase_total_cost=0.0100,
    )
    avg_line = result.splitlines()[3]
    assert "0.0050" in avg_line


def test_averages_row_shows_avg_duration():
    # phase_n=2, phase_elapsed=10.0 → avg 5.0s
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        phase_label="1", phase_n=2, phase_elapsed=10.0,
        phase_total_in=200, phase_total_out=100, phase_total_cost=0.0004,
    )
    avg_line = result.splitlines()[3]
    assert "5.0s" in avg_line


# ── removed / renamed fields ───────────────────────────────────────────────────

def test_request_count_not_shown_separately():
    # Redundant with iteration number in header — removed
    tracker = _session_tracker(4, started_at=0.0, now=40.0)
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        session_tracker=tracker, now=40.0,
    )
    assert "requests" not in result


def test_instantaneous_rpm_not_shown_in_current_iter_row():
    tracker = _session_tracker(1, started_at=0.0, now=6.0)
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        session_tracker=tracker, now=6.0,
        iter_duration=6.0,
    )
    current_iter_line = result.splitlines()[1]
    assert "rpm" not in current_iter_line


def test_in_minute_window_count_not_shown():
    tracker = _session_tracker(3, started_at=0.0, now=30.0)
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        session_tracker=tracker, now=30.0,
    )
    assert "/min" not in result or "llm calls/min" in result


# ── box line count and structure ───────────────────────────────────────────────

def test_box_has_3_lines_without_phase_or_session():
    # top + iter content + bottom
    result = _format_iter_stats(100, 50, "test-model", _MOCK_PRICING)
    assert len(result.splitlines()) == 3


def test_box_has_5_lines_with_session_only():
    # top + iter + session hdr + session content + bottom
    tracker = _session_tracker(1, started_at=0.0, now=10.0)
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        session_tracker=tracker, now=10.0,
    )
    assert len(result.splitlines()) == 5


def test_box_has_5_lines_with_phase_only():
    # top + iter + avg hdr + avg content + bottom
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        phase_label="1", phase_n=1, phase_elapsed=5.0,
        phase_total_in=100, phase_total_out=50, phase_total_cost=0.0002,
    )
    assert len(result.splitlines()) == 5


def test_box_has_7_lines_with_both_phase_and_session():
    # top + iter + avg hdr + avg + session hdr + session + bottom
    tracker = _session_tracker(1, started_at=0.0, now=10.0)
    result = _format_iter_stats(
        100, 50, "test-model", _MOCK_PRICING,
        phase_label="1", phase_n=1, phase_elapsed=10.0,
        phase_total_in=100, phase_total_out=50, phase_total_cost=0.0002,
        session_tracker=tracker, now=10.0,
    )
    lines = result.splitlines()
    assert len(lines) == 7
    assert "averages" in lines[2]
    assert "session" in lines[4]


# ── all lines in box have equal width ─────────────────────────────────────────

def test_all_box_lines_have_equal_width_full_box():
    tracker = _session_tracker(3, started_at=0.0, now=90.0)
    result = _format_iter_stats(
        1500, 800, "test-model", _MOCK_PRICING,
        phase_label="1 (exploration)", phase_n=3, phase_elapsed=90.0,
        phase_total_in=4500, phase_total_out=2400, phase_total_cost=0.0089,
        session_tracker=tracker, now=90.0,
        session_total_in=4500, session_total_out=2400, session_total_cost=0.0089,
        iter_duration=12.5,
    )
    lines = result.splitlines()
    assert len(set(len(line) for line in lines)) == 1, (
        f"Box lines have unequal widths: {[len(l) for l in lines]}"
    )


# ── _format_phase_summary ─────────────────────────────────────────────────────

def test_phase_summary_returns_string():
    result = _format_phase_summary(
        phase_label="1 (exploration)",
        phase_n=10, phase_elapsed=120.0,
        phase_total_in=50000, phase_total_out=2000, phase_total_cost=0.0540,
    )
    assert isinstance(result, str)


def test_phase_summary_includes_phase_label():
    # "1 (exploration)" is rendered as "phase 1 — exploration"
    result = _format_phase_summary(
        phase_label="1 (exploration)",
        phase_n=5, phase_elapsed=60.0,
        phase_total_in=5000, phase_total_out=500, phase_total_cost=0.0060,
    )
    assert "phase 1" in result
    assert "exploration" in result


def test_phase_summary_includes_iter_count():
    result = _format_phase_summary(
        phase_label="1",
        phase_n=31, phase_elapsed=300.0,
        phase_total_in=31000, phase_total_out=3100, phase_total_cost=0.0370,
    )
    assert "31" in result


def test_phase_summary_includes_avg_tokens():
    # phase_n=2, phase_total_in=1000 → avg 500
    result = _format_phase_summary(
        phase_label="1",
        phase_n=2, phase_elapsed=20.0,
        phase_total_in=1000, phase_total_out=200, phase_total_cost=0.0010,
    )
    assert "500" in result


def test_phase_summary_includes_avg_cost():
    # phase_n=4, phase_total_cost=0.0200 → avg $0.0050
    result = _format_phase_summary(
        phase_label="1",
        phase_n=4, phase_elapsed=40.0,
        phase_total_in=400, phase_total_out=200, phase_total_cost=0.0200,
    )
    assert "0.0050" in result
