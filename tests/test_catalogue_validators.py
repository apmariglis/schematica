"""
Tests for Pydantic field validators on MeasurableMetric and QueryableFact.

strip_trailing_semicolon must:
  - Strip a trailing semicolon from SQL strings
  - Strip surrounding whitespace
  - Handle SQL with no semicolon unchanged
  - Reject non-string values with a clear validation error

These are data quality validators — the catalogue is built from LLM output
which often appends semicolons that would break engine.execute() calls.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from schematica.catalogue import MeasurableMetric, QueryableFact


VALID_METRIC = {
    "name": "monthly_revenue",
    "description": "Monthly revenue",
    "sql": "SELECT dt, val FROM t",
    "time_range": {"start": "2024-01-01", "end": "2024-12-31"},
    "granularity": "monthly",
    "unit": "€",
    "tables_used": ["t"],
    "confidence": "high",
    "agent_notes": "",
}

VALID_FACT = {
    "name": "region_lookup",
    "description": "Region reference table",
    "sql": "SELECT * FROM regions",
    "tables_used": ["regions"],
    "agent_notes": "",
}


# ── MeasurableMetric.sql ──────────────────────────────────────────────────────

def test_metric_sql_trailing_semicolon_is_stripped():
    m = MeasurableMetric.model_validate({**VALID_METRIC, "sql": "SELECT dt, val FROM t;"})
    assert m.sql == "SELECT dt, val FROM t"


def test_metric_sql_trailing_whitespace_is_stripped():
    m = MeasurableMetric.model_validate({**VALID_METRIC, "sql": "  SELECT dt, val FROM t  "})
    assert m.sql == "SELECT dt, val FROM t"


def test_metric_sql_semicolon_and_whitespace_both_stripped():
    m = MeasurableMetric.model_validate({**VALID_METRIC, "sql": "  SELECT dt, val FROM t ;  "})
    assert m.sql == "SELECT dt, val FROM t"


def test_metric_sql_without_semicolon_is_unchanged():
    sql = "SELECT dt, val FROM t"
    m = MeasurableMetric.model_validate({**VALID_METRIC, "sql": sql})
    assert m.sql == sql


def test_metric_sql_non_string_raises_validation_error():
    with pytest.raises(ValidationError):
        MeasurableMetric.model_validate({**VALID_METRIC, "sql": 42})


# ── QueryableFact.sql ─────────────────────────────────────────────────────────

def test_fact_sql_trailing_semicolon_is_stripped():
    f = QueryableFact.model_validate({**VALID_FACT, "sql": "SELECT * FROM regions;"})
    assert f.sql == "SELECT * FROM regions"


def test_fact_sql_without_semicolon_is_unchanged():
    sql = "SELECT * FROM regions"
    f = QueryableFact.model_validate({**VALID_FACT, "sql": sql})
    assert f.sql == sql


def test_fact_sql_non_string_raises_validation_error():
    with pytest.raises(ValidationError):
        QueryableFact.model_validate({**VALID_FACT, "sql": None})
