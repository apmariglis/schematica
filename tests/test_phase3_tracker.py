"""
Tests that the accumulated stats row appears in Phase 3 (refinement) iterations.

Phase 3 uses a fresh backend but should share the same _RequestTracker as
Phases 1 and 2, so the accumulated token/time row is visible throughout the
whole run — not just in Phase 1.

Root cause: _run_phase3 did not accept or forward the tracker, so _run_phase
was always called with tracker=None in Phase 3, suppressing the accumulated row.
"""
from __future__ import annotations

from schematica.agent import _format_iter_stats, _RequestTracker


# ── accumulated row requires tracker ─────────────────────────────────────────

def test_accumulated_row_absent_without_tracker():
    result = _format_iter_stats(1000, 200, "test-model", {"test-model": {"input": 1.0, "output": 2.0}})

    assert "accumulated" not in result


def test_accumulated_row_present_with_tracker():
    tracker = _RequestTracker(started_at=0.0)
    tracker.record(now=10.0)

    result = _format_iter_stats(
        1000, 200, "test-model",
        {"test-model": {"input": 1.0, "output": 2.0}},
        tracker=tracker, now=10.0,
    )

    assert "accumulated" in result


# ── tracker persists across phases (shared object, not reset) ─────────────────

def test_tracker_accumulates_across_multiple_record_calls():
    # Simulates Phase 1 recording 3 iters, then Phase 3 recording 2 more.
    # The same tracker instance must hold all 5 records.
    tracker = _RequestTracker(started_at=0.0)

    for t in [10.0, 20.0, 30.0]:       # Phase 1
        tracker.record(now=t)
    for t in [40.0, 50.0]:             # Phase 3
        tracker.record(now=t)

    assert tracker.total == 5


def test_tracker_iter_per_min_reflects_all_phases():
    tracker = _RequestTracker(started_at=0.0)

    for t in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]:
        tracker.record(now=t)

    # 6 records over 60s → 6.0 iter/min
    result = _format_iter_stats(
        500, 100, "test-model",
        {"test-model": {"input": 1.0, "output": 2.0}},
        tracker=tracker, now=60.0,
    )

    assert "6.0 iter/min" in result
