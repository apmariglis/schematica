"""
Tests for _has_temporal_column — detects date/time columns in a table dict.

Covers both typed databases (DATE, TIMESTAMP etc.) and SQLite-style TEXT
columns whose min value is an ISO date string.
"""
from __future__ import annotations

import pytest

from schematica.agent import _has_temporal_column


def _col(name: str, col_type: str, min_val=None) -> dict:
    return {"name": name, "type": col_type, "stats": {"min": min_val}}


def _table(*cols: dict) -> dict:
    return {"columns": list(cols)}


# ── typed date/time columns ───────────────────────────────────────────────────

def test_date_type_is_temporal():
    assert _has_temporal_column(_table(_col("created_at", "DATE")))


def test_timestamp_type_is_temporal():
    assert _has_temporal_column(_table(_col("created_at", "TIMESTAMP")))


def test_datetime_type_is_temporal():
    assert _has_temporal_column(_table(_col("ts", "DATETIME")))


def test_time_type_is_temporal():
    assert _has_temporal_column(_table(_col("t", "TIME")))


def test_type_check_is_case_insensitive():
    assert _has_temporal_column(_table(_col("created_at", "date")))


def test_type_containing_date_keyword_is_temporal():
    # e.g. "TIMESTAMP WITHOUT TIME ZONE"
    assert _has_temporal_column(_table(_col("ts", "TIMESTAMP WITHOUT TIME ZONE")))


# ── SQLite TEXT columns detected by ISO min value ─────────────────────────────

def test_text_column_with_iso_date_min_is_temporal():
    assert _has_temporal_column(_table(_col("opened_at", "TEXT", "2022-01-15")))


def test_text_column_with_iso_datetime_min_is_temporal():
    assert _has_temporal_column(_table(_col("ts", "TEXT", "2022-01-15 08:30:00")))


def test_text_column_with_non_date_min_is_not_temporal():
    assert not _has_temporal_column(_table(_col("name", "TEXT", "Alice")))


def test_text_column_with_no_min_is_not_temporal():
    assert not _has_temporal_column(_table(_col("name", "TEXT", None)))


# ── non-temporal columns ──────────────────────────────────────────────────────

def test_integer_column_is_not_temporal():
    assert not _has_temporal_column(_table(_col("id", "INTEGER")))


def test_real_column_is_not_temporal():
    assert not _has_temporal_column(_table(_col("amount", "REAL")))


def test_boolean_column_is_not_temporal():
    assert not _has_temporal_column(_table(_col("active", "BOOLEAN")))


# ── mixed tables ──────────────────────────────────────────────────────────────

def test_table_with_one_date_column_among_many_is_temporal():
    assert _has_temporal_column(_table(
        _col("id", "INTEGER"),
        _col("name", "TEXT"),
        _col("created_at", "DATE"),
        _col("amount", "REAL"),
    ))


def test_table_with_no_date_columns_is_not_temporal():
    assert not _has_temporal_column(_table(
        _col("id", "INTEGER"),
        _col("name", "TEXT"),
        _col("code", "VARCHAR"),
    ))


def test_empty_table_is_not_temporal():
    assert not _has_temporal_column({"columns": []})


def test_table_with_no_columns_key_is_not_temporal():
    assert not _has_temporal_column({})


# ── lookup table classification (integration-style) ───────────────────────────
# Verifies the classification logic used in run() to build lookup_tables
# and required_tables from a snapshot.

def test_ref_table_without_dates_is_lookup():
    ref_table = _table(
        _col("industry_id", "INTEGER", 1),
        _col("industry_name", "TEXT", "fintech"),
    )
    assert not _has_temporal_column(ref_table)


def test_fact_table_with_date_is_required():
    fact_table = _table(
        _col("account_id", "INTEGER", 1),
        _col("signup_date", "TEXT", "2022-01-01"),
        _col("arr", "REAL", 0.0),
    )
    assert _has_temporal_column(fact_table)


def test_sparse_table_with_date_is_temporal_not_lookup():
    # escalations: only 18 rows but has date columns — should NOT be a lookup
    escalations = _table(
        _col("escalation_id", "INTEGER", 1),
        _col("escalated_at", "TEXT", "2024-06-09"),
        _col("resolved_at", "TEXT", "2024-06-09"),
    )
    assert _has_temporal_column(escalations)
