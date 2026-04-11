"""
Tests for _drop_broken_sql — the SQL syntax check that runs after the agent
submits a catalogue.

Covers:
  - valid metrics and facts are kept
  - metrics with broken SQL are silently dropped (with a console warning)
  - facts with broken SQL are silently dropped
  - when ALL metrics are broken the returned catalogue has an empty metrics list
  - mixed: some valid, some broken — only valid ones survive
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from schematica.agent import _drop_broken_sql
from schematica.catalogue import DataCatalogue, MeasurableMetric, QueryableFact, TimeRange


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(text(
            "CREATE TABLE events (id INTEGER PRIMARY KEY, occurred_at TEXT, magnitude REAL)"
        ))
        conn.execute(text(
            "INSERT INTO events VALUES (1, '2024-01-01', 3.5), (2, '2024-02-01', 7.1)"
        ))
    return eng


def _make_catalogue(metrics=None, facts=None) -> DataCatalogue:
    return DataCatalogue(
        connection="sqlite:///:memory:",
        dialect="sqlite",
        description="test",
        tables=[],
        measurable_metrics=metrics or [],
        queryable_facts=facts or [],
        time_coverage={"start": "2024-01-01", "end": "2024-12-31"},
        data_quality_notes=[],
    )


def _valid_metric(name="magnitude_over_time") -> MeasurableMetric:
    return MeasurableMetric(
        name=name,
        description="Magnitude over time",
        sql="SELECT occurred_at, magnitude FROM events ORDER BY occurred_at",
        time_range=TimeRange(start="2024-01-01", end="2024-12-31"),
        granularity="monthly",
        unit="count",
        tables_used=["events"],
        confidence="high",
        agent_notes="Direct column read",
    )


def _broken_metric(name="broken") -> MeasurableMetric:
    return MeasurableMetric(
        name=name,
        description="Uses non-existent column",
        sql="SELECT occurred_at, nonexistent_column FROM events ORDER BY occurred_at",
        time_range=TimeRange(start="2024-01-01", end="2024-12-31"),
        granularity="monthly",
        unit="count",
        tables_used=["events"],
        confidence="low",
        agent_notes="Broken",
    )


def _valid_fact(name="event_lookup") -> QueryableFact:
    return QueryableFact(
        name=name,
        description="All events",
        sql="SELECT id, occurred_at FROM events",
        tables_used=["events"],
        agent_notes="Direct read",
    )


def _broken_fact(name="broken_fact") -> QueryableFact:
    return QueryableFact(
        name=name,
        description="Broken fact",
        sql="SELECT id, ghost_column FROM events",
        tables_used=["events"],
        agent_notes="Broken",
    )


# ── valid SQL is kept ──────────────────────────────────────────────────────────

def test_valid_metric_is_kept(engine):
    catalogue = _make_catalogue(metrics=[_valid_metric()])

    result = _drop_broken_sql(catalogue, engine)

    assert len(result.measurable_metrics) == 1
    assert result.measurable_metrics[0].name == "magnitude_over_time"


def test_valid_fact_is_kept(engine):
    catalogue = _make_catalogue(facts=[_valid_fact()])

    result = _drop_broken_sql(catalogue, engine)

    assert len(result.queryable_facts) == 1


# ── broken SQL is dropped ──────────────────────────────────────────────────────

def test_broken_metric_is_dropped(engine):
    catalogue = _make_catalogue(metrics=[_broken_metric()])

    result = _drop_broken_sql(catalogue, engine)

    assert len(result.measurable_metrics) == 0


def test_broken_fact_is_dropped(engine):
    catalogue = _make_catalogue(facts=[_broken_fact()])

    result = _drop_broken_sql(catalogue, engine)

    assert len(result.queryable_facts) == 0


# ── all broken → empty catalogue ──────────────────────────────────────────────

def test_all_broken_metrics_produces_empty_metrics_list(engine):
    catalogue = _make_catalogue(metrics=[_broken_metric("a"), _broken_metric("b")])

    result = _drop_broken_sql(catalogue, engine)

    assert result.measurable_metrics == []


# ── mixed valid and broken ─────────────────────────────────────────────────────

def test_only_valid_metrics_survive_when_mixed(engine):
    catalogue = _make_catalogue(metrics=[_valid_metric("good"), _broken_metric("bad")])

    result = _drop_broken_sql(catalogue, engine)

    assert len(result.measurable_metrics) == 1
    assert result.measurable_metrics[0].name == "good"


def test_only_valid_facts_survive_when_mixed(engine):
    catalogue = _make_catalogue(facts=[_valid_fact("good"), _broken_fact("bad")])

    result = _drop_broken_sql(catalogue, engine)

    assert len(result.queryable_facts) == 1
    assert result.queryable_facts[0].name == "good"
