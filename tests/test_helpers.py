"""
Tests for pure helper functions scattered across agent.py and compare_catalogues.py.

No database or LLM required — all functions are deterministic and side-effect-free.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest

from schematica.agent import _tables_referenced_in_sql
from schematica.agent import _tables_used_violations
from schematica.cli import _to_connection_string


# ── _to_connection_string (cli.py) ────────────────────────────────────────────

def test_to_connection_string_passes_through_sqlite_url():
    assert _to_connection_string("sqlite:///data/events.db") == "sqlite:///data/events.db"


def test_to_connection_string_passes_through_postgresql_url():
    assert _to_connection_string("postgresql://user:pw@host/db") == "postgresql://user:pw@host/db"


def test_to_connection_string_passes_through_mysql_url():
    assert _to_connection_string("mysql://user:pw@host/db") == "mysql://user:pw@host/db"


def test_to_connection_string_converts_plain_file_path_to_sqlite():
    assert _to_connection_string("data/events.db") == "sqlite:///data/events.db"


def test_to_connection_string_converts_absolute_path_to_sqlite():
    assert _to_connection_string("/var/data/events.db") == "sqlite:////var/data/events.db"


# ── _db_stem (compare_catalogues.py) ──────────────────────────────────────────
# _get_db_stem uses importlib to dynamically load an external script file by
# path. This cannot be a module-level import because the target is not a
# package — it must be loaded on demand via spec_from_file_location.

def _get_db_stem():
    spec = importlib.util.spec_from_file_location(
        "compare_catalogues",
        pathlib.Path(__file__).parents[1] / "scripts" / "compare_catalogues.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._db_stem


def test_db_stem_extracts_stem_from_file_path():
    fn = _get_db_stem()

    assert fn("data/events.db") == "events"


def test_db_stem_extracts_stem_from_absolute_file_path():
    fn = _get_db_stem()

    assert fn("/var/data/solar_wind.db") == "solar_wind"


def test_db_stem_extracts_stem_from_sqlite_connection_string():
    fn = _get_db_stem()

    assert fn("sqlite:///data/events.db") == "events"


def test_db_stem_extracts_stem_from_postgresql_connection_string():
    fn = _get_db_stem()

    assert fn("postgresql://user:pw@host/mydb") == "mydb"


def test_db_stem_extracts_stem_ignoring_query_params():
    fn = _get_db_stem()

    assert fn("sqlite:///data/events.db?timeout=30") == "events"


# ── _tables_referenced_in_sql (agent.py) ──────────────────────────────────────

def test_tables_referenced_finds_single_from_clause():
    result = _tables_referenced_in_sql("SELECT x FROM events")

    assert result == {"events"}


def test_tables_referenced_finds_join_table():
    result = _tables_referenced_in_sql("SELECT e.x, r.y FROM events e JOIN regions r ON e.id = r.event_id")

    assert "events" in result
    assert "regions" in result


def test_tables_referenced_handles_double_quoted_table_names():
    result = _tables_referenced_in_sql('SELECT x FROM "Event Details"')

    # Names are normalised to lowercase for case-insensitive comparison
    assert "event details" in result


def test_tables_referenced_is_case_insensitive_for_keywords():
    result = _tables_referenced_in_sql("select x from events")

    assert "events" in result


def test_tables_referenced_returns_empty_set_for_no_from_clause():
    result = _tables_referenced_in_sql("SELECT 1")

    assert result == set()


def test_tables_referenced_does_not_crash_on_subquery_expression():
    # FROM (...) produces a match where all four capture groups are empty;
    # the function must not raise StopIteration / RuntimeError.
    result = _tables_referenced_in_sql("SELECT * FROM (SELECT x FROM readings) AS sub")

    # The inner table is captured; the subquery itself is not treated as a name
    assert "readings" in result


# ── _tables_used_violations (agent.py) ────────────────────────────────────────

def test_tables_used_violations_returns_empty_when_all_tables_are_referenced():
    items = [{"name": "m", "sql": "SELECT x FROM events", "tables_used": ["events"]}]

    result = _tables_used_violations(items)

    assert result == []


def test_tables_used_violations_flags_table_listed_but_not_in_sql():
    items = [{"name": "m", "sql": "SELECT x FROM events", "tables_used": ["events", "ghost"]}]

    result = _tables_used_violations(items)

    assert len(result) == 1
    assert "ghost" in result[0]
    assert "m" in result[0]


def test_tables_used_violations_returns_empty_when_tables_used_is_empty():
    items = [{"name": "m", "sql": "SELECT x FROM events", "tables_used": []}]

    result = _tables_used_violations(items)

    assert result == []


def test_tables_used_violations_handles_multiple_items_independently():
    items = [
        {"name": "ok",  "sql": "SELECT x FROM events",  "tables_used": ["events"]},
        {"name": "bad", "sql": "SELECT x FROM readings", "tables_used": ["readings", "phantom"]},
    ]

    result = _tables_used_violations(items)

    assert len(result) == 1
    assert "bad" in result[0]
