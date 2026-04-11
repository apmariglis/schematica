"""
Tests for fuzzy match warning in DataAccessBroker.fetch().

When an exact metric name is given, no warning should be emitted.
When a fuzzy match is used (score < 1.0), a UserWarning must be raised
so the caller knows the requested name was resolved to a different metric.
"""
from __future__ import annotations

import warnings

import pytest
from sqlalchemy import create_engine, text

from schematica.broker import DataAccessBroker


CATALOGUE = {
    "connection": "sqlite:///:memory:",
    "measurable_metrics": [
        {
            "name": "monthly_revenue",
            "description": "Monthly revenue",
            "sql": "SELECT dt, val FROM t ORDER BY dt",
            "time_range": {"start": "2024-01-01", "end": "2024-12-31"},
            "granularity": "monthly",
            "unit": "€",
            "tables_used": ["t"],
            "confidence": "high",
            "agent_notes": "",
        }
    ],
    "queryable_facts": [],
}


@pytest.fixture()
def broker(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE t (dt TEXT, val REAL)"))
        conn.execute(text("INSERT INTO t VALUES ('2024-01-01', 1.0), ('2024-02-01', 2.0)"))
    import json
    catalogue_path = tmp_path / "catalogue.json"
    catalogue_path.write_text(json.dumps(CATALOGUE))
    return DataAccessBroker(str(catalogue_path), f"sqlite:///{db_path}")


# ── exact match — no warning ───────────────────────────────────────────────────

def test_exact_match_emits_no_warning(broker):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning → test failure
        broker.fetch("monthly_revenue")  # exact name


# ── fuzzy match — UserWarning ─────────────────────────────────────────────────

def test_fuzzy_match_emits_user_warning(broker):
    with pytest.warns(UserWarning, match="monthly_revenue"):
        broker.fetch("monthly_revenues")  # close but not exact


def test_fuzzy_match_warning_mentions_requested_name(broker):
    with pytest.warns(UserWarning, match="monthly_revenues"):
        broker.fetch("monthly_revenues")


def test_fuzzy_match_still_returns_correct_data(broker):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = broker.fetch("monthly_revenues")

    assert len(df) == 2
    assert list(df.columns) == ["date", "value"]
