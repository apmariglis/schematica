"""
Tests for _write_output — catalogue JSON and overview Markdown are both written.

The overview file must:
  - Be named <db_stem>_overview_<n>.md (same index as the catalogue JSON)
  - Contain the catalogue's overview text
  - Be written in the same directory as the catalogue JSON
  - Not be written (or written empty) when overview is absent
"""
from __future__ import annotations

import json

import pytest

from schematica.agent import _write_output
from schematica.catalogue import DataCatalogue, MeasurableMetric, QueryableFact, TableSummary, TimeRange


def _make_catalogue(overview: str = "") -> DataCatalogue:
    return DataCatalogue(
        connection="sqlite:///test.db",
        dialect="sqlite",
        description="Test catalogue",
        overview=overview,
        tables=[TableSummary(name="t", row_count=10, description="test", key_columns=["dt"])],
        measurable_metrics=[
            MeasurableMetric(
                name="m", description="d",
                sql="SELECT dt, val FROM t",
                time_range=TimeRange(start="2024-01-01", end="2024-12-31"),
                granularity="monthly", unit="count",
                tables_used=["t"], confidence="high", agent_notes="",
            )
        ],
        queryable_facts=[],
        time_coverage=TimeRange(start="2024-01-01", end="2024-12-31"),
        data_quality_notes=[],
    )


def test_catalogue_json_is_written(tmp_path):
    # out_path is a pattern (no index, no extension) — _write_output appends _1.json
    out = str(tmp_path / "solar_wind_catalogue")
    _write_output(_make_catalogue(), out)
    assert (tmp_path / "solar_wind_catalogue_1.json").exists()


def test_overview_md_is_written_alongside_json(tmp_path):
    out = str(tmp_path / "solar_wind_catalogue")
    _write_output(_make_catalogue(overview="This database tracks solar assets."), out)
    assert (tmp_path / "solar_wind_overview_1.md").exists()


def test_overview_md_contains_overview_text(tmp_path):
    out = str(tmp_path / "mydb_catalogue")
    _write_output(_make_catalogue(overview="Detailed overview text here."), out)
    content = (tmp_path / "mydb_overview_1.md").read_text()
    assert "Detailed overview text here." in content


def test_overview_md_not_written_when_overview_is_empty(tmp_path):
    out = str(tmp_path / "mydb_catalogue")
    _write_output(_make_catalogue(overview=""), out)
    assert not (tmp_path / "mydb_overview_1.md").exists()
