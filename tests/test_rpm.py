"""
Tests for _calc_rpm and _RequestTracker.

_calc_rpm: pure helper, tested exhaustively.

_RequestTracker: stateful — tracks total requests and a per-minute window
counter that resets when 60 seconds have elapsed since the window started.
"""
from __future__ import annotations

import pytest

from schematica.agent import _calc_rpm, _RequestTracker


# ── _calc_rpm ─────────────────────────────────────────────────────────────────

def test_one_request_in_sixty_seconds_is_one_rpm():
    assert _calc_rpm(n_requests=1, elapsed_secs=60.0) == pytest.approx(1.0)


def test_ten_requests_in_two_minutes_is_five_rpm():
    assert _calc_rpm(n_requests=10, elapsed_secs=120.0) == pytest.approx(5.0)


def test_six_requests_in_one_minute_is_six_rpm():
    assert _calc_rpm(n_requests=6, elapsed_secs=60.0) == pytest.approx(6.0)


def test_zero_requests_returns_zero():
    assert _calc_rpm(n_requests=0, elapsed_secs=60.0) == 0.0


def test_zero_elapsed_returns_zero():
    assert _calc_rpm(n_requests=5, elapsed_secs=0.0) == 0.0


def test_negative_elapsed_returns_zero():
    assert _calc_rpm(n_requests=3, elapsed_secs=-1.0) == 0.0


# ── _RequestTracker ───────────────────────────────────────────────────────────

def test_total_starts_at_zero():
    tracker = _RequestTracker(started_at=0.0)
    assert tracker.total == 0


def test_in_minute_starts_at_zero():
    tracker = _RequestTracker(started_at=0.0)
    assert tracker.in_minute == 0


def test_record_increments_total():
    tracker = _RequestTracker(started_at=0.0)
    tracker.record(now=10.0)
    assert tracker.total == 1


def test_record_twice_increments_total_twice():
    tracker = _RequestTracker(started_at=0.0)
    tracker.record(now=10.0)
    tracker.record(now=20.0)
    assert tracker.total == 2


def test_record_increments_in_minute():
    tracker = _RequestTracker(started_at=0.0)
    tracker.record(now=10.0)
    tracker.record(now=20.0)
    assert tracker.in_minute == 2


def test_in_minute_resets_when_sixty_seconds_elapse():
    # Two requests in minute 1, one in minute 2 — in_minute should show 1
    tracker = _RequestTracker(started_at=0.0)
    tracker.record(now=10.0)
    tracker.record(now=20.0)
    tracker.record(now=65.0)   # crosses 60s boundary → new minute
    assert tracker.in_minute == 1


def test_total_does_not_reset_across_minute_boundary():
    tracker = _RequestTracker(started_at=0.0)
    tracker.record(now=10.0)
    tracker.record(now=65.0)
    assert tracker.total == 2


def test_in_minute_resets_again_on_second_minute_boundary():
    tracker = _RequestTracker(started_at=0.0)
    tracker.record(now=10.0)   # minute 1
    tracker.record(now=65.0)   # minute 2
    tracker.record(now=130.0)  # minute 3
    assert tracker.in_minute == 1


def test_multiple_requests_in_second_minute_counted_correctly():
    tracker = _RequestTracker(started_at=0.0)
    tracker.record(now=10.0)   # minute 1
    tracker.record(now=65.0)   # minute 2, req 1
    tracker.record(now=80.0)   # minute 2, req 2
    assert tracker.in_minute == 2


def test_no_reset_just_before_boundary():
    # 59.9 seconds — still within the same minute window
    tracker = _RequestTracker(started_at=0.0)
    tracker.record(now=10.0)
    tracker.record(now=59.9)
    assert tracker.in_minute == 2
