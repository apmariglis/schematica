"""
Tests for pure helper functions scattered across agent.py and compare_catalogues.py.

No database or LLM required — all functions are deterministic and side-effect-free.
"""
from __future__ import annotations

import sys
import types

import pytest


# ── _to_connection_string (agent.py) ──────────────────────────────────────────
# Import only the helper, not the whole agent module (avoids dotenv / LLM setup)

def _get_to_connection_string():
    """Lazy import so we only pull in the helper, not the full agent bootstrap."""
    from schematica.agent import _to_connection_string
    return _to_connection_string


def test_to_connection_string_passes_through_sqlite_url():
    fn = _get_to_connection_string()

    assert fn("sqlite:///data/events.db") == "sqlite:///data/events.db"


def test_to_connection_string_passes_through_postgresql_url():
    fn = _get_to_connection_string()

    assert fn("postgresql://user:pw@host/db") == "postgresql://user:pw@host/db"


def test_to_connection_string_passes_through_mysql_url():
    fn = _get_to_connection_string()

    assert fn("mysql://user:pw@host/db") == "mysql://user:pw@host/db"


def test_to_connection_string_converts_plain_file_path_to_sqlite():
    fn = _get_to_connection_string()

    assert fn("data/events.db") == "sqlite:///data/events.db"


def test_to_connection_string_converts_absolute_path_to_sqlite():
    fn = _get_to_connection_string()

    assert fn("/var/data/events.db") == "sqlite:////var/data/events.db"


# ── _db_stem (compare_catalogues.py) ──────────────────────────────────────────

def _get_db_stem():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "compare_catalogues",
        pathlib.Path(__file__).parents[1] / "scripts" / "compare_catalogues.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Provide a stub for schematica.eval so the import doesn't fail on missing deps
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

def _get_tables_referenced():
    from schematica.agent import _tables_referenced_in_sql
    return _tables_referenced_in_sql


def test_tables_referenced_finds_single_from_clause():
    fn = _get_tables_referenced()

    result = fn("SELECT x FROM events")

    assert result == {"events"}


def test_tables_referenced_finds_join_table():
    fn = _get_tables_referenced()

    result = fn("SELECT e.x, r.y FROM events e JOIN regions r ON e.id = r.event_id")

    assert "events" in result
    assert "regions" in result


def test_tables_referenced_handles_double_quoted_table_names():
    fn = _get_tables_referenced()

    result = fn('SELECT x FROM "Event Details"')

    # Names are normalised to lowercase for case-insensitive comparison
    assert "event details" in result


def test_tables_referenced_is_case_insensitive_for_keywords():
    fn = _get_tables_referenced()

    result = fn("select x from events")

    assert "events" in result


def test_tables_referenced_returns_empty_set_for_no_from_clause():
    fn = _get_tables_referenced()

    result = fn("SELECT 1")

    assert result == set()


# ── _tables_used_violations (agent.py) ────────────────────────────────────────

def _get_tables_used_violations():
    from schematica.agent import _tables_used_violations
    return _tables_used_violations


def test_tables_used_violations_returns_empty_when_all_tables_are_referenced():
    fn = _get_tables_used_violations()
    items = [{"name": "m", "sql": "SELECT x FROM events", "tables_used": ["events"]}]

    result = fn(items)

    assert result == []


def test_tables_used_violations_flags_table_listed_but_not_in_sql():
    fn = _get_tables_used_violations()
    items = [{"name": "m", "sql": "SELECT x FROM events", "tables_used": ["events", "ghost"]}]

    result = fn(items)

    assert len(result) == 1
    assert "ghost" in result[0]
    assert "m" in result[0]


def test_tables_used_violations_returns_empty_when_tables_used_is_empty():
    fn = _get_tables_used_violations()
    items = [{"name": "m", "sql": "SELECT x FROM events", "tables_used": []}]

    result = fn(items)

    assert result == []


def test_tables_used_violations_handles_multiple_items_independently():
    fn = _get_tables_used_violations()
    items = [
        {"name": "ok",  "sql": "SELECT x FROM events",  "tables_used": ["events"]},
        {"name": "bad", "sql": "SELECT x FROM readings", "tables_used": ["readings", "phantom"]},
    ]

    result = fn(items)

    assert len(result) == 1
    assert "bad" in result[0]
