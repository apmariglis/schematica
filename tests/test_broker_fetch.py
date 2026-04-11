"""
Tests for DataAccessBroker.fetch() — core data retrieval behaviour.

Covers:
  - Returned DataFrame shape and column names
  - Date range filtering (start_date / end_date)
  - NULL values in the value column are dropped
  - df.attrs metadata is populated
  - KeyError raised for unknown metric
"""
from __future__ import annotations

import json
import warnings

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from schematica.broker import DataAccessBroker


CATALOGUE = {
    "connection": "sqlite:///:memory:",
    "measurable_metrics": [
        {
            "name": "monthly_revenue",
            "description": "Monthly revenue",
            "sql": "SELECT dt, val FROM revenue ORDER BY dt",
            "time_range": {"start": "2024-01-01", "end": "2024-06-01"},
            "granularity": "monthly",
            "unit": "€",
            "tables_used": ["revenue"],
            "confidence": "high",
            "agent_notes": "test metric",
        }
    ],
    "queryable_facts": [],
}

ROWS = [
    ("2024-01", 100.0),
    ("2024-02", 200.0),
    ("2024-03", None),   # NULL — should be dropped
    ("2024-04", 400.0),
    ("2024-05", 500.0),
]


@pytest.fixture()
def broker(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE revenue (dt TEXT, val REAL)"))
        for dt, val in ROWS:
            conn.execute(text("INSERT INTO revenue VALUES (:dt, :val)"), {"dt": dt, "val": val})

    catalogue_path = tmp_path / "catalogue.json"
    catalogue_path.write_text(json.dumps(CATALOGUE))
    return DataAccessBroker(str(catalogue_path), f"sqlite:///{db_path}")


# ── basic shape ───────────────────────────────────────────────────────────────

def test_fetch_returns_dataframe(broker):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = broker.fetch("monthly_revenue")

    assert isinstance(df, pd.DataFrame)


def test_fetch_has_date_and_value_columns(broker):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = broker.fetch("monthly_revenue")

    assert list(df.columns) == ["date", "value"]


def test_fetch_value_column_is_numeric(broker):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = broker.fetch("monthly_revenue")

    assert pd.api.types.is_numeric_dtype(df["value"])


# ── NULL handling ─────────────────────────────────────────────────────────────

def test_fetch_drops_null_value_rows(broker):
    # The fixture has one NULL in the value column; it must be excluded.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = broker.fetch("monthly_revenue")

    assert df["value"].isna().sum() == 0


def test_fetch_row_count_excludes_nulls(broker):
    # 5 rows inserted, 1 NULL → 4 rows expected
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = broker.fetch("monthly_revenue")

    assert len(df) == 4


# ── date range filtering ──────────────────────────────────────────────────────

def test_fetch_start_date_excludes_earlier_rows(broker):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = broker.fetch("monthly_revenue", start_date="2024-03-01")

    # 2024-01 and 2024-02 must be absent; 2024-03 is NULL so also absent
    assert all(row >= "2024-03" for row in df["date"])
    assert "2024-01" not in df["date"].values
    assert "2024-02" not in df["date"].values


def test_fetch_end_date_excludes_later_rows(broker):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = broker.fetch("monthly_revenue", end_date="2024-03-31")

    assert all(row <= "2024-03" for row in df["date"])
    assert "2024-04" not in df["date"].values
    assert "2024-05" not in df["date"].values


def test_fetch_with_both_dates_filters_to_window(broker):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = broker.fetch("monthly_revenue", start_date="2024-02-01", end_date="2024-04-30")

    dates = set(df["date"])
    assert "2024-01" not in dates
    assert "2024-05" not in dates
    assert "2024-04" in dates


def test_fetch_result_is_sorted_by_date(broker):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = broker.fetch("monthly_revenue")

    assert list(df["date"]) == sorted(df["date"])


# ── attrs metadata ────────────────────────────────────────────────────────────

def test_fetch_attrs_contains_metric_name(broker):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = broker.fetch("monthly_revenue")

    assert df.attrs["metric_name"] == "monthly_revenue"


def test_fetch_attrs_contains_granularity(broker):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = broker.fetch("monthly_revenue")

    assert df.attrs["granularity"] == "monthly"


def test_fetch_attrs_contains_unit(broker):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = broker.fetch("monthly_revenue")

    assert df.attrs["unit"] == "€"


# ── error handling ────────────────────────────────────────────────────────────

def test_fetch_raises_key_error_for_unknown_metric(broker):
    with pytest.raises(KeyError):
        broker.fetch("this_metric_does_not_exist_at_all")
