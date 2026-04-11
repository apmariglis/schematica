"""
Tests for the Phase 3 refinement fallback guard.

When the refinement agent returns a catalogue that ends up empty after
_drop_broken_sql, the system must fall back to the pre-refinement
(directly-patched) catalogue rather than returning empty output.

We test _select_phase3_result — a pure helper that encodes this decision —
rather than the full _run_phase3 function (which requires a live LLM).
"""
from __future__ import annotations

import pytest

from schematica.agent import _select_phase3_result
from schematica.catalogue import DataCatalogue, MeasurableMetric, TimeRange


def _make_catalogue(n_metrics: int, name_prefix: str = "m") -> DataCatalogue:
    metrics = [
        MeasurableMetric(
            name=f"{name_prefix}_{i}",
            description="test",
            sql=f"SELECT period, val_{i} FROM t ORDER BY period",
            time_range=TimeRange(start="2024-01-01", end="2024-12-31"),
            granularity="monthly",
            unit="count",
            tables_used=["t"],
            confidence="high",
            agent_notes="test",
        )
        for i in range(n_metrics)
    ]
    return DataCatalogue(
        connection="sqlite:///:memory:",
        dialect="sqlite",
        description="test",
        tables=[],
        measurable_metrics=metrics,
        queryable_facts=[],
        time_coverage={"start": "2024-01-01", "end": "2024-12-31"},
        data_quality_notes=[],
    )


# ── empty refined catalogue → fall back ───────────────────────────────────────

def test_falls_back_to_patched_when_refined_is_empty():
    patched   = _make_catalogue(n_metrics=5, name_prefix="patched")
    refined   = _make_catalogue(n_metrics=0)

    result, did_fallback = _select_phase3_result(refined, patched)

    assert did_fallback is True
    assert len(result.measurable_metrics) == 5
    assert result.measurable_metrics[0].name.startswith("patched")


# ── refined has metrics → use it ──────────────────────────────────────────────

def test_uses_refined_when_it_has_metrics():
    patched = _make_catalogue(n_metrics=5, name_prefix="patched")
    refined = _make_catalogue(n_metrics=4, name_prefix="refined")

    result, did_fallback = _select_phase3_result(refined, patched)

    assert did_fallback is False
    assert result.measurable_metrics[0].name.startswith("refined")


# ── heavy loss → warns but still uses refined ─────────────────────────────────

def test_does_not_fall_back_when_refined_has_some_metrics():
    # Even if 60% of metrics were dropped, as long as refined is non-empty
    # we should use it (with a warning) rather than silently discard the fixes.
    patched = _make_catalogue(n_metrics=10, name_prefix="patched")
    refined = _make_catalogue(n_metrics=4,  name_prefix="refined")

    result, did_fallback = _select_phase3_result(refined, patched)

    assert did_fallback is False
    assert len(result.measurable_metrics) == 4
