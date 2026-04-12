"""
Tests for _build_catalogue — catalogue construction from raw agent output.

When optional fields (description, queryable_facts, data_quality_notes) are
submitted as null by the LLM, _build_catalogue must fall back to the empty
default rather than passing None to Pydantic (which would raise a confusing
validation error deep in the stack).
"""
from __future__ import annotations

import pytest

from schematica.agent import _build_catalogue


SNAPSHOT = {
    "connection_string": "sqlite:///test.db",
    "dialect": "sqlite",
}

VALID_DATA = {
    "description": "A test catalogue",
    "tables": [{"name": "t", "row_count": 10, "description": "test table", "key_columns": ["dt", "val"]}],
    "measurable_metrics": [
        {
            "name": "monthly_count",
            "description": "Monthly count",
            "sql": "SELECT dt, COUNT(*) FROM t GROUP BY dt",
            "time_range": {"start": "2024-01-01", "end": "2024-12-31"},
            "granularity": "monthly",
            "unit": "count",
            "tables_used": ["t"],
            "confidence": "high",
            "agent_notes": "",
        }
    ],
    "queryable_facts": [],
    "time_coverage": {"start": "2024-01-01", "end": "2024-12-31"},
    "data_quality_notes": [],
}


# ── happy path ────────────────────────────────────────────────────────────────

def test_build_catalogue_succeeds_with_valid_data():
    catalogue = _build_catalogue(VALID_DATA, SNAPSHOT)
    assert catalogue.description == "A test catalogue"
    assert len(catalogue.measurable_metrics) == 1


# ── null optional fields fall back to defaults ────────────────────────────────

def test_null_description_falls_back_to_empty_string():
    # LLM submitted description: null — must not raise, must default to ""
    data = {**VALID_DATA, "description": None}
    catalogue = _build_catalogue(data, SNAPSHOT)
    assert catalogue.description == ""


def test_null_queryable_facts_falls_back_to_empty_list():
    data = {**VALID_DATA, "queryable_facts": None}
    catalogue = _build_catalogue(data, SNAPSHOT)
    assert catalogue.queryable_facts == []


def test_null_data_quality_notes_falls_back_to_empty_list():
    data = {**VALID_DATA, "data_quality_notes": None}
    catalogue = _build_catalogue(data, SNAPSHOT)
    assert catalogue.data_quality_notes == []


# ── absent optional fields still fall back to defaults ───────────────────────

def test_absent_description_falls_back_to_empty_string():
    data = {k: v for k, v in VALID_DATA.items() if k != "description"}
    catalogue = _build_catalogue(data, SNAPSHOT)
    assert catalogue.description == ""


def test_absent_queryable_facts_falls_back_to_empty_list():
    data = {k: v for k, v in VALID_DATA.items() if k != "queryable_facts"}
    catalogue = _build_catalogue(data, SNAPSHOT)
    assert catalogue.queryable_facts == []
