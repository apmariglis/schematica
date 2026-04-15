"""
Tests that evaluator crashes are excluded from the Phase 3 agent_issues list.

When eval.py's analysis framework crashes (e.g. 'No module named numpy.rec')
sql_ok=True but the error starts with "eval error:". These are infrastructure
failures — the SQL is fine but the eval environment is broken. Sending them to
the Phase 3 refinement agent wastes budget because the agent cannot fix an
environment issue.

The fix: filter agent_issues through _is_evaluator_crash before building the
refinement prompt.

Tested via the public filter helper _filter_agent_issues which encapsulates
the issue-selection logic so it can be unit-tested without running a full phase.
"""
from __future__ import annotations

from types import SimpleNamespace

from schematica.agent import _filter_agent_issues
from schematica.eval import MetricResult, FactResult


def _metric(name: str, sql_ok: bool, error: str) -> MetricResult:
    r = MetricResult(
        name=name, confidence="high", granularity="monthly",
        unit="count", declared_start="2023-01-01", declared_end="2023-12-01",
    )
    r.sql_ok = sql_ok
    r.error  = error
    r.status = "FAIL" if (not sql_ok or error) else "PASS"
    # Give it a time_range attribute so hasattr check works
    return r


def _fact(name: str, sql_ok: bool, error: str) -> FactResult:
    r = FactResult(name=name)
    r.sql_ok = sql_ok
    r.error  = error
    r.status = "FAIL" if error else "PASS"
    return r


def _mock_metric_item(name: str) -> object:
    """Minimal metric item with time_range so hasattr(item, 'time_range') is True."""
    return SimpleNamespace(
        name=name,
        sql="SELECT 1",
        time_range=SimpleNamespace(start="2023-01-01", end="2023-12-01"),
    )


def _mock_fact_item(name: str) -> object:
    """Minimal fact item without time_range."""
    return SimpleNamespace(name=name, sql="SELECT 1")


# ── evaluator crashes are excluded ────────────────────────────────────────────

def test_evaluator_crash_metric_excluded_from_agent_issues():
    item = _mock_metric_item("monthly_leads")
    result = _metric("monthly_leads", sql_ok=True, error="eval error: No module named 'numpy.rec'")

    issues = _filter_agent_issues([(item, result)], [])

    assert len(issues) == 0


def test_evaluator_crash_fact_excluded_from_agent_issues():
    item = _mock_fact_item("region_lookup")
    result = _fact("region_lookup", sql_ok=True, error="eval error: No module named 'numpy.rec'")

    issues = _filter_agent_issues([], [(item, result)])

    assert len(issues) == 0


def test_all_evaluator_crashes_returns_empty_list():
    metrics = [(_mock_metric_item(f"m{i}"),
                _metric(f"m{i}", sql_ok=True, error="eval error: something broke"))
               for i in range(5)]

    issues = _filter_agent_issues(metrics, [])

    assert issues == []


# ── real SQL failures still included ──────────────────────────────────────────

def test_sql_error_metric_included_in_agent_issues():
    item = _mock_metric_item("broken_metric")
    result = _metric("broken_metric", sql_ok=False, error="no such table: foo")

    issues = _filter_agent_issues([(item, result)], [])

    assert len(issues) == 1


def test_warn_metric_included_in_agent_issues():
    item = _mock_metric_item("sparse_metric")
    result = _metric("sparse_metric", sql_ok=True, error="sparse")

    issues = _filter_agent_issues([(item, result)], [])

    assert len(issues) == 1


def test_date_mismatch_metric_excluded_by_direct_patch_logic():
    # date_mismatch is patched directly, not via agent — should not appear
    item = _mock_metric_item("date_metric")
    result = _metric("date_metric", sql_ok=True, error="date_mismatch")

    issues = _filter_agent_issues([(item, result)], [])

    assert len(issues) == 0


def test_mixed_list_only_returns_real_failures():
    crash_item   = _mock_metric_item("crash")
    crash_result = _metric("crash", sql_ok=True, error="eval error: something")

    sql_item   = _mock_metric_item("sql_fail")
    sql_result = _metric("sql_fail", sql_ok=False, error="no such column: x")

    issues = _filter_agent_issues(
        [(crash_item, crash_result), (sql_item, sql_result)], []
    )

    assert len(issues) == 1
    assert issues[0][0].name == "sql_fail"
