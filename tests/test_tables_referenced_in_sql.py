"""
Tests for _tables_referenced_in_sql.

The function must extract the table name (not schema prefix) from all
FROM/JOIN clauses, including:
  - bare identifiers: FROM orders
  - backtick-quoted:  FROM `order_details`
  - bracket-quoted:   FROM [Order Details]
  - double-quoted:    FROM "Order Details"
  - schema-qualified: FROM public.orders  →  should return "orders"
  - quoted schema:    FROM "public"."orders"  →  should return "orders"

Missing qualified-name handling caused _tables_used_violations to falsely
flag metrics that referenced schema-qualified tables.
"""
from __future__ import annotations

import pytest

from schematica.agent import _tables_referenced_in_sql


# ── bare identifiers ──────────────────────────────────────────────────────────

def test_bare_from_clause():
    assert "orders" in _tables_referenced_in_sql("SELECT * FROM orders")


def test_bare_join_clause():
    sql = "SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id"
    result = _tables_referenced_in_sql(sql)
    assert "orders" in result
    assert "customers" in result


# ── quoted identifiers ────────────────────────────────────────────────────────

def test_double_quoted_table_name():
    assert "order details" in _tables_referenced_in_sql('SELECT * FROM "Order Details"')


def test_backtick_quoted_table_name():
    assert "order_details" in _tables_referenced_in_sql("SELECT * FROM `order_details`")


def test_bracket_quoted_table_name():
    assert "order details" in _tables_referenced_in_sql("SELECT * FROM [Order Details]")


# ── schema-qualified names ────────────────────────────────────────────────────

def test_schema_qualified_bare_extracts_table_not_schema():
    # "public.orders" should resolve to "orders", not "public"
    result = _tables_referenced_in_sql("SELECT * FROM public.orders")
    assert "orders" in result
    assert "public" not in result


def test_schema_qualified_join_extracts_table_not_schema():
    sql = "SELECT * FROM public.orders JOIN public.customers ON 1=1"
    result = _tables_referenced_in_sql(sql)
    assert "orders" in result
    assert "customers" in result
    assert "public" not in result


def test_double_quoted_schema_and_table():
    # "public"."orders" — both parts quoted
    result = _tables_referenced_in_sql('SELECT * FROM "public"."orders"')
    assert "orders" in result
    assert "public" not in result


# ── subqueries should not produce false positives ─────────────────────────────

def test_subquery_alias_is_not_included():
    sql = "SELECT * FROM (SELECT id FROM orders) AS sub"
    result = _tables_referenced_in_sql(sql)
    assert "sub" not in result
    assert "orders" in result
