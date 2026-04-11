"""
Tests for _execute_query — the SQL execution gate used during Phase 1 exploration.

These tests focus on the security boundary: only SELECT statements should reach
the database. Any DDL or DML must be rejected before execution.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text

from schematica.agent import _execute_query


@pytest.fixture()
def engine():
    """In-memory SQLite with a simple table so SELECT queries have something to hit."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE events (id INTEGER PRIMARY KEY, label TEXT)"))
        conn.execute(text("INSERT INTO events VALUES (1, 'alpha'), (2, 'beta')"))
    return eng


# ── allowed queries ────────────────────────────────────────────────────────────

def test_execute_query_returns_results_for_valid_select(engine):
    result = _execute_query(engine, "SELECT id, label FROM events", "test")

    assert "alpha" in result
    assert "ERROR" not in result


def test_execute_query_allows_select_with_leading_line_comment(engine):
    sql = "-- fetch all\nSELECT id, label FROM events"

    result = _execute_query(engine, sql, "test")

    assert "ERROR" not in result


# ── blocked statements ─────────────────────────────────────────────────────────

def test_execute_query_rejects_drop_table(engine):
    result = _execute_query(engine, "DROP TABLE events", "test")

    assert result.startswith("ERROR")
    # Table must still exist
    assert "events" in inspect(engine).get_table_names()


def test_execute_query_rejects_delete(engine):
    result = _execute_query(engine, "DELETE FROM events", "test")

    assert result.startswith("ERROR")


def test_execute_query_rejects_insert(engine):
    result = _execute_query(engine, "INSERT INTO events VALUES (3, 'gamma')", "test")

    assert result.startswith("ERROR")


def test_execute_query_rejects_multi_statement_that_starts_with_select(engine):
    # SELECT passes the first-token check but the second statement is destructive.
    # The guard must detect the semicolon and reject the whole input.
    sql = "SELECT 1; DROP TABLE events"

    result = _execute_query(engine, sql, "test")

    assert result.startswith("ERROR")
    assert "events" in inspect(engine).get_table_names()


def test_execute_query_rejects_block_comment_followed_by_drop(engine):
    sql = "/* looks innocent */ DROP TABLE events"

    result = _execute_query(engine, sql, "test")

    assert result.startswith("ERROR")
    assert "events" in inspect(engine).get_table_names()
