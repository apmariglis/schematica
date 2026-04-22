"""
Tests for _pre_validate_catalogue_items — per-item Pydantic pre-validation
run inside _run_phase before accepting a finish_catalogue submission.

This catches schema errors (e.g. missing fields) and returns them to the
agent as rejection feedback rather than letting them crash in _build_catalogue.

Covers: measurable_metrics, queryable_facts, key_terms, table_relationships.
"""
from __future__ import annotations

import pytest

from schematica.agent import _pre_validate_catalogue_items


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
    "group": "Revenue",
}

VALID_FACT = {
    "name": "region_lookup",
    "description": "Region reference table",
    "sql": "SELECT * FROM regions",
    "tables_used": ["regions"],
    "agent_notes": "",
}

VALID_KEY_TERM = {
    "term": "Basket Size",
    "definition": "The average number of tracks per invoice transaction.",
}

VALID_TABLE_REL = {
    "table_a": "Invoice",
    "table_b": "Customer",
    "join_key": "CustomerId",
}


# ── happy path ────────────────────────────────────────────────────────────────

def test_returns_empty_list_when_all_items_valid():
    data = {
        "measurable_metrics": [VALID_METRIC],
        "queryable_facts": [VALID_FACT],
        "key_terms": [VALID_KEY_TERM],
        "table_relationships": [VALID_TABLE_REL],
    }

    result = _pre_validate_catalogue_items(data)

    assert result == []


def test_returns_empty_list_for_empty_data():
    result = _pre_validate_catalogue_items({})

    assert result == []


def test_returns_empty_list_when_all_lists_empty():
    data = {
        "measurable_metrics": [],
        "queryable_facts": [],
        "key_terms": [],
        "table_relationships": [],
    }

    result = _pre_validate_catalogue_items(data)

    assert result == []


# ── measurable_metrics errors ─────────────────────────────────────────────────

def test_catches_metric_missing_required_field():
    # A metric without 'sql' should produce an error
    bad_metric = {k: v for k, v in VALID_METRIC.items() if k != "sql"}
    data = {"measurable_metrics": [bad_metric]}

    errors = _pre_validate_catalogue_items(data)

    assert len(errors) == 1
    assert "measurable_metrics[0]" in errors[0]
    assert "monthly_revenue" in errors[0]


def test_metric_error_message_includes_index_and_name():
    bad_metric = {k: v for k, v in VALID_METRIC.items() if k != "granularity"}
    data = {"measurable_metrics": [VALID_METRIC, bad_metric]}

    errors = _pre_validate_catalogue_items(data)

    assert any("measurable_metrics[1]" in e for e in errors)


def test_skips_non_dict_metric_entries():
    # Non-dict entries are caught by earlier checks — skip silently here
    data = {"measurable_metrics": ["bare_string", VALID_METRIC]}

    errors = _pre_validate_catalogue_items(data)

    assert errors == []


# ── queryable_facts errors ────────────────────────────────────────────────────

def test_catches_fact_missing_required_field():
    bad_fact = {k: v for k, v in VALID_FACT.items() if k != "sql"}
    data = {"queryable_facts": [bad_fact]}

    errors = _pre_validate_catalogue_items(data)

    assert len(errors) == 1
    assert "queryable_facts[0]" in errors[0]
    assert "region_lookup" in errors[0]


# ── key_terms errors ──────────────────────────────────────────────────────────

def test_catches_key_term_missing_definition():
    # This is the exact failure mode observed in the Chinook run:
    # LLM submitted {"term": "Basket Size"} with no "definition" key.
    bad_term = {"term": "Basket Size"}
    data = {"key_terms": [bad_term]}

    errors = _pre_validate_catalogue_items(data)

    assert len(errors) == 1
    assert "key_terms[0]" in errors[0]
    assert "Basket Size" in errors[0]


def test_catches_key_term_missing_term():
    bad_term = {"definition": "some definition"}
    data = {"key_terms": [bad_term]}

    errors = _pre_validate_catalogue_items(data)

    assert len(errors) == 1
    assert "key_terms[0]" in errors[0]


def test_valid_key_term_produces_no_error():
    data = {"key_terms": [VALID_KEY_TERM]}

    errors = _pre_validate_catalogue_items(data)

    assert errors == []


def test_error_index_points_to_correct_key_term():
    # First term is valid, second is broken — error must say [1]
    bad_term = {"term": "Churn"}  # missing definition
    data = {"key_terms": [VALID_KEY_TERM, bad_term]}

    errors = _pre_validate_catalogue_items(data)

    assert any("key_terms[1]" in e for e in errors)
    assert not any("key_terms[0]" in e for e in errors)


# ── table_relationships errors ────────────────────────────────────────────────

def test_catches_table_relationship_missing_join_key():
    bad_rel = {"table_a": "Invoice", "table_b": "Customer"}  # missing join_key
    data = {"table_relationships": [bad_rel]}

    errors = _pre_validate_catalogue_items(data)

    assert len(errors) == 1
    assert "table_relationships[0]" in errors[0]


def test_valid_table_relationship_produces_no_error():
    data = {"table_relationships": [VALID_TABLE_REL]}

    errors = _pre_validate_catalogue_items(data)

    assert errors == []


# ── multiple errors ───────────────────────────────────────────────────────────

def test_collects_errors_from_multiple_item_types():
    # One bad metric + one bad key_term — both errors should be returned
    bad_metric = {k: v for k, v in VALID_METRIC.items() if k != "sql"}
    bad_term = {"term": "MRR"}  # missing definition

    data = {
        "measurable_metrics": [bad_metric],
        "key_terms": [bad_term],
    }

    errors = _pre_validate_catalogue_items(data)

    assert len(errors) == 2
    assert any("measurable_metrics" in e for e in errors)
    assert any("key_terms" in e for e in errors)
