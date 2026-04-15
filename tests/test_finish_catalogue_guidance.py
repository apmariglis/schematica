"""
Tests that finish_catalogue's tool schema and rejection messages give
weak models (e.g. Llama) enough context to submit the correct format.

Two failure modes observed with Llama 3.3 70B:
  1. tables submitted as bare strings instead of full objects
  2. model never self-corrects because rejection message lacks a paste-ready example

Option A fix: add a concrete example to the `tables` JSON Schema description —
seen before the model generates any output, prevents the error at source.

Option B fix: rejection message includes a copy-paste-ready object example —
gives the model a template to follow when correcting its mistake.
"""
from __future__ import annotations

from schematica.agent import (
    _FINISH_CATALOGUE_TOOL,
    _BARE_TABLES_ERROR_MSG,
    _TABLES_NOT_LIST_ERROR_MSG,
    _FK_REJECTION_MSG,
)


# ── Option A: tool schema description for tables ─────────────────────────────

def test_tables_property_has_description():
    tables_schema = _FINISH_CATALOGUE_TOOL["input_schema"]["properties"]["tables"]

    assert "description" in tables_schema


def test_tables_description_is_nonempty():
    desc = _FINISH_CATALOGUE_TOOL["input_schema"]["properties"]["tables"]["description"]

    assert desc.strip()


def test_tables_description_mentions_full_object_not_string():
    desc = _FINISH_CATALOGUE_TOOL["input_schema"]["properties"]["tables"]["description"]

    # Must warn the model not to send bare strings
    text = desc.lower()
    assert "not a string" in text or "full object" in text or "not strings" in text


def test_tables_description_shows_name_field_in_example():
    desc = _FINISH_CATALOGUE_TOOL["input_schema"]["properties"]["tables"]["description"]

    assert '"name"' in desc


def test_tables_description_shows_row_count_field_in_example():
    desc = _FINISH_CATALOGUE_TOOL["input_schema"]["properties"]["tables"]["description"]

    assert '"row_count"' in desc


def test_tables_description_shows_key_columns_field_in_example():
    desc = _FINISH_CATALOGUE_TOOL["input_schema"]["properties"]["tables"]["description"]

    assert '"key_columns"' in desc


def test_tables_description_example_is_an_object_literal():
    # Must contain { ... } to show it's an object, not just name the fields
    desc = _FINISH_CATALOGUE_TOOL["input_schema"]["properties"]["tables"]["description"]

    assert "{" in desc and "}" in desc


# ── Option B: rejection message contains a concrete paste-ready example ──────

def test_bare_tables_rejection_message_is_nonempty():
    assert _BARE_TABLES_ERROR_MSG.strip()


def test_bare_tables_rejection_message_shows_name_field():
    assert '"name"' in _BARE_TABLES_ERROR_MSG


def test_bare_tables_rejection_message_shows_row_count_field():
    assert '"row_count"' in _BARE_TABLES_ERROR_MSG


def test_bare_tables_rejection_message_shows_key_columns_field():
    assert '"key_columns"' in _BARE_TABLES_ERROR_MSG


def test_bare_tables_rejection_message_contains_object_literal():
    # The example must show a { ... } object, not just name the required fields
    assert "{" in _BARE_TABLES_ERROR_MSG and "}" in _BARE_TABLES_ERROR_MSG


def test_bare_tables_rejection_message_explains_what_went_wrong():
    # Must tell the model it sent a string where an object was expected
    text = _BARE_TABLES_ERROR_MSG.lower()
    assert "string" in text


# ── _TABLES_NOT_LIST_ERROR_MSG: tables submitted as a non-list type ───────────
# Separate from bare-string check: this fires when `tables` is not a list
# at all (e.g. the whole catalogue JSON was passed as a single string).

def test_tables_not_list_error_msg_is_nonempty():
    assert _TABLES_NOT_LIST_ERROR_MSG.strip()


def test_tables_not_list_error_msg_mentions_list():
    assert "list" in _TABLES_NOT_LIST_ERROR_MSG.lower()


def test_tables_not_list_error_msg_mentions_string():
    # Must tell the model what it actually sent
    assert "string" in _TABLES_NOT_LIST_ERROR_MSG.lower()


def test_tables_not_list_error_msg_shows_correct_shape():
    # Must show the agent the expected format — a list of objects
    assert "[{" in _TABLES_NOT_LIST_ERROR_MSG or "list of" in _TABLES_NOT_LIST_ERROR_MSG.lower()


# ── _FK_REJECTION_MSG: actionable guidance when FK pairs are uncovered ────────
# The message is a format-string with {pairs_str} placeholder.

def test_fk_rejection_msg_is_nonempty():
    assert _FK_REJECTION_MSG.strip()


def test_fk_rejection_msg_contains_pairs_placeholder():
    assert "{pairs_str}" in _FK_REJECTION_MSG


def test_fk_rejection_msg_tells_agent_to_look_at_run_queries():
    # Agent must be directed to its own prior run_query calls as the source
    text = _FK_REJECTION_MSG.lower()
    assert "run_query" in text


def test_fk_rejection_msg_requires_measurable_metric_not_fact():
    text = _FK_REJECTION_MSG.lower()
    assert "measurable_metric" in text or "measurable_metrics" in text
    assert "queryable_fact" in text or "fact" in text
