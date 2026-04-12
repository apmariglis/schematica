"""
Tests for _uncovered_fk_pairs — FK relationship coverage check.

When a database has FK relationships, the catalogue should include at least
one metric whose SQL JOINs both tables in the relationship. This function
detects which FK pairs have no covering metric.

This prevents the agent silently skipping cross-table analysis when FKs exist.
"""
from __future__ import annotations

import pytest

from schematica.agent import _uncovered_fk_pairs


# ── no FKs — nothing to check ─────────────────────────────────────────────────

def test_no_fk_pairs_returns_empty():
    metrics = [{"name": "m1", "sql": "SELECT dt, val FROM orders"}]
    result = _uncovered_fk_pairs(metrics, fk_pairs=[])
    assert result == []


# ── FK pair covered by a metric ───────────────────────────────────────────────

def test_fk_covered_by_join_metric_returns_empty():
    metrics = [
        {
            "name": "orders_per_customer_monthly",
            "sql": "SELECT dt, COUNT(*) FROM orders JOIN customers ON orders.customer_id = customers.id GROUP BY dt",
        }
    ]
    result = _uncovered_fk_pairs(metrics, fk_pairs=[("orders", "customers")])
    assert result == []


def test_fk_covered_regardless_of_direction():
    # FK declared as orders → customers; metric has customers JOIN orders — still covered.
    metrics = [
        {
            "name": "m",
            "sql": "SELECT dt, COUNT(*) FROM customers JOIN orders ON customers.id = orders.customer_id GROUP BY dt",
        }
    ]
    result = _uncovered_fk_pairs(metrics, fk_pairs=[("orders", "customers")])
    assert result == []


# ── FK pair not covered ───────────────────────────────────────────────────────

def test_single_uncovered_fk_pair_is_returned():
    metrics = [
        {"name": "order_count", "sql": "SELECT dt, COUNT(*) FROM orders GROUP BY dt"},
    ]
    result = _uncovered_fk_pairs(metrics, fk_pairs=[("orders", "customers")])
    assert len(result) == 1
    assert set(result[0]) == {"orders", "customers"}


def test_multiple_fk_pairs_with_one_uncovered():
    metrics = [
        {
            "name": "orders_per_customer",
            "sql": "SELECT dt, COUNT(*) FROM orders JOIN customers ON 1=1 GROUP BY dt",
        }
    ]
    fk_pairs = [
        ("orders", "customers"),
        ("orders", "products"),
    ]
    result = _uncovered_fk_pairs(metrics, fk_pairs=fk_pairs)
    assert len(result) == 1
    assert set(result[0]) == {"orders", "products"}


def test_all_fk_pairs_covered():
    metrics = [
        {
            "name": "m1",
            "sql": "SELECT dt, COUNT(*) FROM orders JOIN customers ON 1=1 GROUP BY dt",
        },
        {
            "name": "m2",
            "sql": "SELECT dt, SUM(qty) FROM orders JOIN products ON 1=1 GROUP BY dt",
        },
    ]
    fk_pairs = [("orders", "customers"), ("orders", "products")]
    result = _uncovered_fk_pairs(metrics, fk_pairs=fk_pairs)
    assert result == []


def test_fact_covering_fk_does_not_count():
    # Only metrics (not facts) cover FKs — facts are excluded from this check.
    # The function only receives metrics, so this is implicitly handled.
    metrics = []   # no metrics at all
    result = _uncovered_fk_pairs(metrics, fk_pairs=[("orders", "customers")])
    assert len(result) == 1


# ── case insensitivity ────────────────────────────────────────────────────────

def test_table_name_comparison_is_case_insensitive():
    metrics = [
        {
            "name": "m",
            "sql": "SELECT dt, COUNT(*) FROM Orders JOIN Customers ON 1=1 GROUP BY dt",
        }
    ]
    result = _uncovered_fk_pairs(metrics, fk_pairs=[("orders", "customers")])
    assert result == []
