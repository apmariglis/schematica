"""
Tests for _is_evaluator_crash — detect when Phase 3 eval failed due to a
framework error (numpy/pandas version incompatibility) rather than a bad SQL query.

When `sql_ok` is True but the error starts with "eval error:", the SQL ran
fine but the DataFrame analysis crashed. Schematica should NOT send these to
the Phase 3 refinement agent — the agent can't fix an environment problem.

Example: "eval error: No module named 'numpy.rec'" (numpy 2.x removed numpy.rec)
"""
from __future__ import annotations

from schematica.eval import MetricResult, FactResult, _is_evaluator_crash


def _metric_result(sql_ok: bool, error: str) -> MetricResult:
    r = MetricResult(
        name="test", confidence="high", granularity="monthly",
        unit="$", declared_start="2023-01-01", declared_end="2023-12-01",
    )
    r.sql_ok = sql_ok
    r.error  = error
    r.status = "FAIL"
    return r


def _fact_result(sql_ok: bool, error: str) -> FactResult:
    r = FactResult(name="test")
    r.sql_ok = sql_ok
    r.error  = error
    r.status = "FAIL"
    return r


# ── detects evaluator crashes correctly ───────────────────────────────────────

def test_sql_ok_with_eval_error_prefix_is_evaluator_crash():
    r = _metric_result(sql_ok=True, error="eval error: No module named 'numpy.rec'")

    assert _is_evaluator_crash(r) is True


def test_sql_ok_with_different_eval_error_is_evaluator_crash():
    r = _metric_result(sql_ok=True, error="eval error: cannot import name 'NaTType'")

    assert _is_evaluator_crash(r) is True


def test_sql_failed_with_eval_error_is_not_evaluator_crash():
    # sql_ok=False means the SQL itself errored — not an evaluator infrastructure issue
    r = _metric_result(sql_ok=False, error="eval error: something weird")

    assert _is_evaluator_crash(r) is False


def test_sql_ok_with_normal_warn_code_is_not_evaluator_crash():
    r = _metric_result(sql_ok=True, error="sparse")

    assert _is_evaluator_crash(r) is False


def test_sql_ok_with_sql_error_is_not_evaluator_crash():
    r = _metric_result(sql_ok=False, error="no such column: foo")

    assert _is_evaluator_crash(r) is False


def test_sql_ok_with_empty_error_is_not_evaluator_crash():
    r = _metric_result(sql_ok=True, error="")

    assert _is_evaluator_crash(r) is False


def test_fact_result_evaluator_crash_detected():
    r = _fact_result(sql_ok=True, error="eval error: No module named 'numpy.rec'")

    assert _is_evaluator_crash(r) is True


def test_fact_result_sql_error_is_not_evaluator_crash():
    r = _fact_result(sql_ok=False, error="no such table: foo")

    assert _is_evaluator_crash(r) is False
