"""
eval.py — Pure computation logic for catalogue quality evaluation.

Extracted from scripts/eval_catalogue.py so the agent loop can evaluate
catalogue entries against the live database without importing Rich or argparse.

Consumers:
  - scripts/eval_catalogue.py  (human-readable CLI report)
  - schematica.agent  (Phase 3 — Refinement)
"""
from __future__ import annotations

import re

import pandas as pd
from sqlalchemy import text

WARN_NULL_RATE      = 0.10   # >10 % nulls → WARN
WARN_MIN_ROWS       = 3      # fewer than 3 rows → WARN
WARN_DATE_PARSE_PCT = 0.20   # >20 % of col-0 values unparseable as dates → WARN


class MetricResult:
    __slots__ = [
        "name", "confidence", "granularity", "unit",
        "sql_ok", "error",
        "n_cols", "col_names",
        "n_rows",
        "null_rate",
        "value_min", "value_max",
        "declared_start", "declared_end",
        "actual_start", "actual_end",
        "date_range_ok",
        "period_boundary_ok",
        "date_col_ok",
        "duplicate_of",
        "status",
    ]

    def __init__(self, name: str, confidence: str, granularity: str, unit: str,
                 declared_start: str, declared_end: str):
        self.name           = name
        self.confidence     = confidence
        self.granularity    = granularity
        self.unit           = unit
        self.declared_start = declared_start
        self.declared_end   = declared_end

        # filled by evaluation
        self.sql_ok      = False
        self.error       = ""
        self.n_cols      = 0
        self.col_names   = []
        self.n_rows      = 0
        self.null_rate   = 0.0
        self.value_min   = None
        self.value_max   = None
        self.actual_start = ""
        self.actual_end   = ""
        self.date_range_ok      = False
        self.period_boundary_ok = True
        self.date_col_ok        = True
        self.duplicate_of       = ""   # name of the first metric with identical SQL, or ""
        self.status      = "FAIL"   # PASS | WARN | FAIL


class FactResult:
    __slots__ = ["name", "sql_ok", "error", "n_cols", "n_rows", "status"]

    def __init__(self, name: str):
        self.name   = name
        self.sql_ok = False
        self.error  = ""
        self.n_cols = 0
        self.n_rows = 0
        self.status = "FAIL"


def evaluate_metric(engine, metric: dict) -> MetricResult:
    result = MetricResult(
        name           = metric["name"],
        confidence     = metric.get("confidence", "?"),
        granularity    = metric.get("granularity", "?"),
        unit           = metric.get("unit", "?"),
        declared_start = metric.get("time_range", {}).get("start", ""),
        declared_end   = metric.get("time_range", {}).get("end", ""),
    )

    sql = metric.get("sql", "").strip()
    if not sql:
        result.error = "No SQL in catalogue entry"
        return result

    # Run the query
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
    except Exception as exc:
        result.error = str(exc)
        return result

    result.sql_ok    = True
    result.n_cols    = len(df.columns)
    result.col_names = list(df.columns)
    result.n_rows    = len(df)

    if result.n_cols < 2:
        result.error = f"Only {result.n_cols} column(s) returned; expected 2"
        return result

    if result.n_rows == 0:
        result.error = "zero_rows"
        result.status = "WARN"
        return result

    # Rename first two cols as date / value for analysis
    df = df.rename(columns={df.columns[0]: "_date", df.columns[1]: "_value"})
    df["_value"] = pd.to_numeric(df["_value"], errors="coerce")

    result.null_rate  = df["_value"].isna().mean()
    result.value_min  = float(df["_value"].min()) if not df["_value"].isna().all() else None
    result.value_max  = float(df["_value"].max()) if not df["_value"].isna().all() else None
    result.actual_start = str(df["_date"].iloc[0])[:10]
    result.actual_end   = str(df["_date"].iloc[-1])[:10]

    # Date column parseability: col 0 should be interpretable as dates.
    # Small integers that pass pd.to_datetime (e.g. Unix timestamps near epoch)
    # are caught by the range check — we only flag if coercion itself fails often.
    parsed_dates = pd.to_datetime(df["_date"], errors="coerce", format="mixed")
    nat_rate = float(parsed_dates.isna().mean())
    result.date_col_ok = nat_rate <= WARN_DATE_PARSE_PCT

    # Date range accuracy: actual should be within ±1 month of declared
    declared_start_prefix = (result.declared_start or "")[:7]
    declared_end_prefix   = (result.declared_end   or "")[:7]
    actual_start_prefix   = result.actual_start[:7]
    actual_end_prefix     = result.actual_end[:7]

    start_ok = (actual_start_prefix >= declared_start_prefix) if declared_start_prefix else True
    end_ok   = (actual_end_prefix   <= declared_end_prefix)   if declared_end_prefix   else True
    result.date_range_ok = start_ok and end_ok

    # Period boundary check: for periodic granularities the declared start/end
    # should align to the period boundary (e.g. monthly → first day of month).
    _BOUNDARY_SUFFIX = {
        "monthly":   "-01",
        "quarterly": "-01",
        "annual":    "-01-01",
    }
    granularity = metric.get("granularity", "")
    suffix = _BOUNDARY_SUFFIX.get(granularity)
    if suffix:
        start_aligned = not result.declared_start or result.declared_start.endswith(suffix)
        end_aligned   = not result.declared_end   or result.declared_end.endswith(suffix)
        result.period_boundary_ok = start_aligned and end_aligned
    else:
        result.period_boundary_ok = True

    # Status
    issues = []
    if result.null_rate > WARN_NULL_RATE:
        issues.append("high_nulls")
    if result.n_rows < WARN_MIN_ROWS:
        issues.append("sparse")
    if not result.date_range_ok:
        issues.append("date_mismatch")
    if result.n_cols != 2:
        issues.append("extra_cols")
    if not result.period_boundary_ok:
        issues.append("period_boundary")
    if not result.date_col_ok:
        issues.append("non_date_col")
    # Constant-value: same value every period → useless as a trend metric
    if (result.n_rows > 1
            and result.value_min is not None
            and result.value_min == result.value_max):
        issues.append("constant_values")

    result.status = "WARN" if issues else "PASS"
    result.error  = ", ".join(issues) if issues else ""

    return result


def evaluate_fact(engine, fact: dict) -> FactResult:
    result = FactResult(name=fact["name"])

    sql = fact.get("sql", "").strip()
    if not sql:
        result.error = "No SQL in catalogue entry"
        return result

    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
    except Exception as exc:
        result.error = str(exc)
        return result

    result.sql_ok = True
    result.n_cols = len(df.columns)
    result.n_rows = len(df)

    if result.n_cols == 0:
        result.error = "Query returned no columns"
        return result

    if result.n_rows == 0:
        result.error = "zero_rows"
        result.status = "WARN"
        return result

    result.status = "PASS"
    return result


def _normalise_sql(sql: str) -> str:
    """Return a canonical form of a SQL string for duplicate detection."""
    return re.sub(r"\s+", " ", sql.strip().lower().rstrip(";").strip())


def check_duplicate_sql(metrics: list[dict]) -> dict[str, str]:
    """
    Return a mapping of metric_name → name_of_first_metric_with_same_sql for
    every metric whose normalised SQL is identical to an earlier one.

    Metrics that appear first with a given SQL are not included in the result.
    """
    seen: dict[str, str] = {}   # normalised_sql → first metric name
    duplicates: dict[str, str] = {}
    for m in metrics:
        name = m.get("name", "")
        sql  = m.get("sql", "")
        if not sql:
            continue
        key = _normalise_sql(sql)
        if key in seen:
            duplicates[name] = seen[key]
        else:
            seen[key] = name
    return duplicates
