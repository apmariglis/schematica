"""
Tests for schematica.eval — catalogue quality evaluation logic.

All tests use an in-memory SQLite engine so no files are needed.
The test database contains a simple sensor readings table:
  readings(recorded_at TEXT, value REAL)
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from schematica.eval import (
    evaluate_metric,
    evaluate_fact,
    check_duplicate_sql,
    _normalise_sql,
)


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def engine():
    """In-memory SQLite with a simple monthly sensor readings table."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(text("""
            CREATE TABLE readings (
                recorded_at TEXT NOT NULL,
                value       REAL NOT NULL
            )
        """))
        conn.execute(text("""
            INSERT INTO readings VALUES
                ('2023-01-01', 10.0),
                ('2023-02-01', 20.0),
                ('2023-03-01', 30.0),
                ('2023-04-01', 40.0)
        """))
    return eng


@pytest.fixture()
def valid_metric():
    """A well-formed metric entry that should PASS."""
    return {
        "name":        "monthly_avg_reading",
        "confidence":  "high",
        "granularity": "monthly",
        "unit":        "units",
        "time_range":  {"start": "2023-01-01", "end": "2023-04-01"},
        "sql":         "SELECT recorded_at, value FROM readings ORDER BY recorded_at",
    }


# ── _normalise_sql ─────────────────────────────────────────────────────────────

def test_normalise_sql_collapses_whitespace():
    result = _normalise_sql("SELECT  a,\n  b  FROM  t")

    assert result == "select a, b from t"


def test_normalise_sql_strips_trailing_semicolon():
    result = _normalise_sql("SELECT a FROM t;")

    assert result == "select a from t"


def test_normalise_sql_lowercases():
    result = _normalise_sql("SELECT A FROM T")

    assert result == "select a from t"


def test_normalise_sql_handles_leading_and_trailing_whitespace():
    result = _normalise_sql("  SELECT a FROM t  ")

    assert result == "select a from t"


# ── check_duplicate_sql ────────────────────────────────────────────────────────

def test_check_duplicate_sql_returns_empty_when_all_unique():
    metrics = [
        {"name": "a", "sql": "SELECT x FROM t"},
        {"name": "b", "sql": "SELECT y FROM t"},
    ]

    result = check_duplicate_sql(metrics)

    assert result == {}


def test_check_duplicate_sql_detects_exact_duplicate():
    metrics = [
        {"name": "first",  "sql": "SELECT x FROM t"},
        {"name": "second", "sql": "SELECT x FROM t"},
    ]

    result = check_duplicate_sql(metrics)

    assert result == {"second": "first"}


def test_check_duplicate_sql_detects_whitespace_normalised_duplicate():
    metrics = [
        {"name": "first",  "sql": "SELECT x FROM t"},
        {"name": "second", "sql": "SELECT  x  FROM  t"},
    ]

    result = check_duplicate_sql(metrics)

    assert result == {"second": "first"}


def test_check_duplicate_sql_detects_case_normalised_duplicate():
    metrics = [
        {"name": "first",  "sql": "select x from t"},
        {"name": "second", "sql": "SELECT x FROM t"},
    ]

    result = check_duplicate_sql(metrics)

    assert result == {"second": "first"}


def test_check_duplicate_sql_only_flags_later_occurrences():
    # Three metrics with the same SQL — only the second and third are flagged
    metrics = [
        {"name": "a", "sql": "SELECT x FROM t"},
        {"name": "b", "sql": "SELECT x FROM t"},
        {"name": "c", "sql": "SELECT x FROM t"},
    ]

    result = check_duplicate_sql(metrics)

    assert "a" not in result
    assert result["b"] == "a"
    assert result["c"] == "a"


def test_check_duplicate_sql_skips_metrics_with_no_sql():
    metrics = [
        {"name": "a", "sql": ""},
        {"name": "b", "sql": ""},
    ]

    result = check_duplicate_sql(metrics)

    assert result == {}


# ── evaluate_metric — happy path ───────────────────────────────────────────────

def test_evaluate_metric_passes_for_valid_query(engine, valid_metric):
    result = evaluate_metric(engine, valid_metric)

    assert result.status == "PASS"


def test_evaluate_metric_sets_sql_ok_on_success(engine, valid_metric):
    result = evaluate_metric(engine, valid_metric)

    assert result.sql_ok is True
    assert result.error == ""


def test_evaluate_metric_reports_correct_row_count(engine, valid_metric):
    result = evaluate_metric(engine, valid_metric)

    assert result.n_rows == 4


def test_evaluate_metric_reports_correct_column_count(engine, valid_metric):
    result = evaluate_metric(engine, valid_metric)

    assert result.n_cols == 2


def test_evaluate_metric_reports_value_range(engine, valid_metric):
    result = evaluate_metric(engine, valid_metric)

    assert result.value_min == pytest.approx(10.0)
    assert result.value_max == pytest.approx(40.0)


def test_evaluate_metric_reports_zero_null_rate(engine, valid_metric):
    result = evaluate_metric(engine, valid_metric)

    assert result.null_rate == pytest.approx(0.0)


def test_evaluate_metric_reports_actual_date_range(engine, valid_metric):
    result = evaluate_metric(engine, valid_metric)

    assert result.actual_start == "2023-01-01"
    assert result.actual_end   == "2023-04-01"


# ── evaluate_metric — SQL failure ──────────────────────────────────────────────

def test_evaluate_metric_fails_for_invalid_sql(engine):
    metric = {
        "name": "bad", "confidence": "high", "granularity": "monthly",
        "unit": "units", "time_range": {},
        "sql": "SELECT * FROM nonexistent_table",
    }

    result = evaluate_metric(engine, metric)

    assert result.status == "FAIL"
    assert result.sql_ok is False


def test_evaluate_metric_fails_when_sql_is_empty(engine):
    metric = {
        "name": "empty", "confidence": "high", "granularity": "monthly",
        "unit": "units", "time_range": {},
        "sql": "",
    }

    result = evaluate_metric(engine, metric)

    assert result.status == "FAIL"
    assert result.sql_ok is False


def test_evaluate_metric_fails_for_single_column_query(engine):
    metric = {
        "name": "one_col", "confidence": "high", "granularity": "monthly",
        "unit": "units", "time_range": {},
        "sql": "SELECT recorded_at FROM readings",
    }

    result = evaluate_metric(engine, metric)

    assert result.status == "FAIL"
    assert result.n_cols == 1


# ── evaluate_metric — WARN conditions ─────────────────────────────────────────

def test_evaluate_metric_warns_for_zero_rows(engine):
    metric = {
        "name": "empty_result", "confidence": "high", "granularity": "monthly",
        "unit": "units", "time_range": {},
        "sql": "SELECT recorded_at, value FROM readings WHERE 1=0",
    }

    result = evaluate_metric(engine, metric)

    assert result.status == "WARN"
    assert "zero_rows" in result.error


def test_evaluate_metric_warns_when_fewer_than_three_rows_returned(engine):
    metric = {
        "name": "sparse", "confidence": "high", "granularity": "monthly",
        "unit": "units", "time_range": {},
        "sql": "SELECT recorded_at, value FROM readings LIMIT 2",
    }

    result = evaluate_metric(engine, metric)

    assert result.status == "WARN"
    assert "sparse" in result.error


def test_evaluate_metric_warns_when_more_than_ten_percent_of_values_are_null():
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE t (dt TEXT, val REAL)"))
        conn.execute(text("""
            INSERT INTO t VALUES
                ('2023-01-01', NULL),
                ('2023-02-01', NULL),
                ('2023-03-01', NULL),
                ('2023-04-01', 1.0)
        """))
    metric = {
        "name": "mostly_null", "confidence": "high", "granularity": "monthly",
        "unit": "units", "time_range": {},
        "sql": "SELECT dt, val FROM t",
    }

    result = evaluate_metric(eng, metric)

    assert result.status == "WARN"
    assert "high_nulls" in result.error


def test_evaluate_metric_warns_when_all_values_are_identical():
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE t (dt TEXT, val REAL)"))
        conn.execute(text("""
            INSERT INTO t VALUES
                ('2023-01-01', 5.0),
                ('2023-02-01', 5.0),
                ('2023-03-01', 5.0)
        """))
    metric = {
        "name": "flat_line", "confidence": "high", "granularity": "monthly",
        "unit": "units", "time_range": {},
        "sql": "SELECT dt, val FROM t",
    }

    result = evaluate_metric(eng, metric)

    assert result.status == "WARN"
    assert "constant_values" in result.error


def test_evaluate_metric_warns_when_actual_range_falls_outside_declared_range(engine, valid_metric):
    # Declared range is entirely in the past — actual data is later
    valid_metric["time_range"] = {"start": "2020-01-01", "end": "2021-12-01"}

    result = evaluate_metric(engine, valid_metric)

    assert result.status == "WARN"
    assert "date_mismatch" in result.error
    assert result.date_range_ok is False


def test_evaluate_metric_warns_for_extra_columns(engine):
    metric = {
        "name": "three_cols", "confidence": "high", "granularity": "monthly",
        "unit": "units", "time_range": {},
        "sql": "SELECT recorded_at, value, 1 AS extra FROM readings",
    }

    result = evaluate_metric(engine, metric)

    assert result.status == "WARN"
    assert "extra_cols" in result.error


def test_evaluate_metric_warns_when_monthly_time_range_does_not_start_on_first_of_month(engine):
    metric = {
        "name": "bad_boundary", "confidence": "high", "granularity": "monthly",
        "unit": "units",
        "time_range": {"start": "2023-01-15", "end": "2023-04-15"},
        "sql": "SELECT recorded_at, value FROM readings",
    }

    result = evaluate_metric(engine, metric)

    assert result.status == "WARN"
    assert "period_boundary" in result.error


# ── evaluate_fact ──────────────────────────────────────────────────────────────

def test_evaluate_fact_passes_for_valid_query(engine):
    fact = {"name": "all_readings", "sql": "SELECT recorded_at, value FROM readings"}

    result = evaluate_fact(engine, fact)

    assert result.status == "PASS"
    assert result.sql_ok is True
    assert result.n_rows == 4
    assert result.n_cols == 2


def test_evaluate_fact_fails_for_invalid_sql(engine):
    fact = {"name": "bad", "sql": "SELECT * FROM no_such_table"}

    result = evaluate_fact(engine, fact)

    assert result.status == "FAIL"
    assert result.sql_ok is False


def test_evaluate_fact_fails_when_sql_is_missing(engine):
    fact = {"name": "empty", "sql": ""}

    result = evaluate_fact(engine, fact)

    assert result.status == "FAIL"


def test_evaluate_fact_warns_for_zero_rows(engine):
    fact = {"name": "no_rows", "sql": "SELECT recorded_at, value FROM readings WHERE 1=0"}

    result = evaluate_fact(engine, fact)

    assert result.status == "WARN"
    assert "zero_rows" in result.error
