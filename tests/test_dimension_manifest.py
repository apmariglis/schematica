"""
Tests for _build_dimension_manifest — deterministic extraction of categorical
dimension values from the schema snapshot for injection into the LLM context.

Behaviours covered:
  - Direct text columns with low cardinality appear in the manifest
  - FK integer columns are dereferenced to the label column of the target table
  - PK columns are excluded (they are identifiers, not breakdown dimensions)
  - Numeric/date columns (min/max stats) are excluded
  - High-cardinality text columns are excluded (too many values to be useful)
  - Unary columns (n_distinct == 1) are excluded (no breakdown is possible)
  - FK pointing to a high-row-count fact table is excluded
  - Values are listed in alphabetical order for reproducibility
  - When no categorical columns exist the function returns an empty string
  - The manifest section header is present when there is content
"""
from __future__ import annotations

import pytest

from schematica.introspect import _build_dimension_manifest, _DIMENSION_CARDINALITY_LIMIT


# ── snapshot builder helpers ──────────────────────────────────────────────────

def _text_col(name: str, values: list[str], pk: bool = False) -> dict:
    return {
        "name": name,
        "type": "TEXT",
        "primary_key": pk,
        "nullable": True,
        "stats": {
            "n_distinct": len(values),
            "top_values": {v: 1 for v in values},
            "n_null": 0,
        },
    }


def _numeric_col(name: str, pk: bool = False) -> dict:
    return {
        "name": name,
        "type": "INTEGER",
        "primary_key": pk,
        "nullable": True,
        "stats": {"min": 1, "max": 100, "n_null": 0},
    }


def _table(
    name: str,
    columns: list[dict],
    fks: list[dict] | None = None,
    row_count: int = 1000,
) -> dict:
    return {
        "name": name,
        "row_count": row_count,
        "columns": columns,
        "foreign_keys": fks or [],
        "sample_rows": [],
    }


def _snapshot(*tables: dict) -> dict:
    return {
        "connection_string": "sqlite:///test.db",
        "dialect": "sqlite",
        "tables": list(tables),
    }


# ── direct text column tests ──────────────────────────────────────────────────

def test_low_cardinality_text_column_appears_in_manifest():
    # A column with a small, known set of values is the canonical breakdown dimension.
    snapshot = _snapshot(
        _table("orders", [_text_col("status", ["open", "closed", "pending"])])
    )

    manifest = _build_dimension_manifest(snapshot)

    assert "orders.status" in manifest
    assert "open" in manifest


def test_values_are_listed_in_alphabetical_order():
    # Sorting makes the output deterministic and diff-friendly.
    snapshot = _snapshot(
        _table("t", [_text_col("region", ["LATAM", "AMER", "EMEA", "APAC"])])
    )

    manifest = _build_dimension_manifest(snapshot)

    idx_apac  = manifest.index("APAC")
    idx_emea  = manifest.index("EMEA")
    idx_latam = manifest.index("LATAM")
    assert idx_apac < idx_emea < idx_latam


def test_column_at_cardinality_limit_is_included():
    # Exactly at the limit must be included (boundary check).
    values = [f"val_{i}" for i in range(_DIMENSION_CARDINALITY_LIMIT)]

    snapshot = _snapshot(
        _table("t", [_text_col("dim", values)])
    )

    manifest = _build_dimension_manifest(snapshot)

    assert "t.dim" in manifest


def test_column_above_cardinality_limit_is_excluded():
    # One value over the limit → too many for a useful breakdown.
    values = [f"val_{i}" for i in range(_DIMENSION_CARDINALITY_LIMIT + 1)]

    snapshot = _snapshot(
        _table("t", [_text_col("category", values)])
    )

    manifest = _build_dimension_manifest(snapshot)

    assert "t.category" not in manifest


def test_pk_column_excluded():
    # PKs are identifiers, not breakdown dimensions.
    snapshot = _snapshot(
        _table("t", [_text_col("code", ["A", "B", "C"], pk=True)])
    )

    manifest = _build_dimension_manifest(snapshot)

    assert "t.code" not in manifest


def test_numeric_column_excluded():
    # Numeric columns have min/max stats rather than top_values; skip them.
    snapshot = _snapshot(
        _table("t", [_numeric_col("amount")])
    )

    manifest = _build_dimension_manifest(snapshot)

    assert "t.amount" not in manifest


def test_unary_column_excluded():
    # n_distinct == 1 means every row has the same value — no breakdown is possible.
    snapshot = _snapshot(
        _table("t", [_text_col("constant", ["always"])])
    )

    manifest = _build_dimension_manifest(snapshot)

    assert "t.constant" not in manifest


# ── FK dereference tests ──────────────────────────────────────────────────────

def test_fk_integer_column_dereferenced_to_label_values():
    # A numeric FK column should be resolved to the human-readable label
    # column of the target lookup table.
    ref_table = _table(
        "ref_plans",
        [
            _numeric_col("plan_id", pk=True),
            _text_col("name", ["Free", "Growth", "Business"]),
        ],
        row_count=3,
    )
    fact_table = _table(
        "accounts",
        [
            _numeric_col("account_id", pk=True),
            _numeric_col("plan_id"),
        ],
        fks=[{"from_cols": ["plan_id"], "to_table": "ref_plans", "to_cols": ["plan_id"]}],
    )

    manifest = _build_dimension_manifest(_snapshot(fact_table, ref_table))

    assert "accounts.plan_id" in manifest
    assert "ref_plans" in manifest
    assert "Business" in manifest
    assert "Free" in manifest
    assert "Growth" in manifest


def test_fk_to_high_row_count_table_excluded():
    # A FK pointing to a large fact table (e.g. accounts, events) is not a
    # useful dimension — the label column would be an identifier, not a category.
    large_table = _table(
        "accounts",
        [
            _numeric_col("account_id", pk=True),
            _text_col("name", [f"Acme {i}" for i in range(20)]),
        ],
        row_count=50_000,
    )
    fact_table = _table(
        "orders",
        [_numeric_col("account_id")],
        fks=[{"from_cols": ["account_id"], "to_table": "accounts", "to_cols": ["account_id"]}],
    )

    manifest = _build_dimension_manifest(_snapshot(fact_table, large_table))

    assert "orders.account_id" not in manifest


def test_fk_label_values_are_also_sorted_alphabetically():
    ref_table = _table(
        "ref_regions",
        [
            _numeric_col("id", pk=True),
            _text_col("name", ["LATAM", "AMER", "EMEA", "APAC"]),
        ],
        row_count=4,
    )
    fact_table = _table(
        "sales",
        [_numeric_col("region_id")],
        fks=[{"from_cols": ["region_id"], "to_table": "ref_regions", "to_cols": ["id"]}],
    )

    manifest = _build_dimension_manifest(_snapshot(fact_table, ref_table))

    idx_apac  = manifest.index("APAC")
    idx_latam = manifest.index("LATAM")
    assert idx_apac < idx_latam


# ── manifest structure ────────────────────────────────────────────────────────

def test_manifest_contains_section_header():
    snapshot = _snapshot(
        _table("t", [_text_col("status", ["open", "closed"])])
    )

    manifest = _build_dimension_manifest(snapshot)

    assert "DIMENSION" in manifest.upper()


def test_empty_snapshot_returns_empty_string():
    manifest = _build_dimension_manifest(_snapshot())

    assert manifest == ""


def test_no_categorical_columns_returns_empty_string():
    snapshot = _snapshot(
        _table("t", [_numeric_col("id", pk=True), _numeric_col("amount")])
    )

    manifest = _build_dimension_manifest(snapshot)

    assert manifest == ""


def test_multiple_tables_all_appear_in_manifest():
    snapshot = _snapshot(
        _table("orders",   [_text_col("status",   ["open", "closed"])]),
        _table("products", [_text_col("category", ["widget", "gadget"])]),
    )

    manifest = _build_dimension_manifest(snapshot)

    assert "orders.status" in manifest
    assert "products.category" in manifest


def test_json_values_column_excluded_from_manifest():
    # Columns whose values look like JSON objects/arrays are not useful breakdown
    # dimensions — they are structured data stored in a text column.
    snapshot = _snapshot(
        _table("t", [_text_col("config", ['{"flag": true}', '{"flag": false}'])])
    )

    manifest = _build_dimension_manifest(snapshot)

    assert "t.config" not in manifest
