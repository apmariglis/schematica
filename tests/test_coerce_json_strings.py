"""
Tests for _coerce_json_strings — normalise finish_catalogue submissions where
the model accidentally JSON-encoded list/dict fields as strings.

This happens when LLMs double-encode JSON:  instead of submitting
  {"tables": [{...}]}
they submit
  {"tables": "[{...}]"}

We detect this pattern and parse it silently so the rest of the pipeline
sees the expected native type.
"""
from __future__ import annotations

import json
import pytest

from schematica.agent import _coerce_json_strings


# ── list fields submitted as JSON strings ─────────────────────────────────────

def test_tables_string_is_parsed_to_list():
    row = {"tables": json.dumps([{"name": "t", "row_count": 5}])}
    result = _coerce_json_strings(row)
    assert isinstance(result["tables"], list)
    assert result["tables"][0]["name"] == "t"


def test_measurable_metrics_string_is_parsed_to_list():
    metrics = [{"name": "m", "sql": "SELECT 1"}]
    row = {"measurable_metrics": json.dumps(metrics)}
    result = _coerce_json_strings(row)
    assert isinstance(result["measurable_metrics"], list)
    assert result["measurable_metrics"][0]["name"] == "m"


def test_queryable_facts_string_is_parsed_to_list():
    facts = [{"name": "f", "sql": "SELECT 1"}]
    row = {"queryable_facts": json.dumps(facts)}
    result = _coerce_json_strings(row)
    assert isinstance(result["queryable_facts"], list)


# ── dict field submitted as JSON string ───────────────────────────────────────

def test_time_coverage_string_is_parsed_to_dict():
    row = {"time_coverage": json.dumps({"start": "2020-01-01", "end": "2024-12-31"})}
    result = _coerce_json_strings(row)
    assert isinstance(result["time_coverage"], dict)
    assert result["time_coverage"]["start"] == "2020-01-01"


# ── already-native types are left untouched ───────────────────────────────────

def test_list_field_already_a_list_is_unchanged():
    tables = [{"name": "t"}]
    row = {"tables": tables}
    result = _coerce_json_strings(row)
    assert result["tables"] is tables


def test_dict_field_already_a_dict_is_unchanged():
    tc = {"start": "2020-01-01", "end": "2024-12-31"}
    row = {"time_coverage": tc}
    result = _coerce_json_strings(row)
    assert result["time_coverage"] is tc


def test_other_fields_are_not_touched():
    row = {"description": "some text", "overview": "more text"}
    result = _coerce_json_strings(row)
    assert result["description"] == "some text"
    assert result["overview"] == "more text"


# ── malformed JSON strings are left as-is (don't crash) ─────────────────────

def test_unparseable_string_is_left_as_is():
    row = {"tables": "not valid json"}
    result = _coerce_json_strings(row)
    assert result["tables"] == "not valid json"


def test_returns_new_dict_not_mutating_input():
    original = {"tables": json.dumps([{"name": "t"}])}
    result = _coerce_json_strings(original)
    # original should not be mutated
    assert isinstance(original["tables"], str)
    assert isinstance(result["tables"], list)
