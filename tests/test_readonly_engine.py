"""
Tests for make_readonly_engine — the read-only engine used during Phase 1 exploration.

The key guarantee: no SQL that could alter the database should ever succeed,
regardless of how the SQL string is constructed.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text

from schematica.db import make_readonly_engine


@pytest.fixture()
def db_path(tmp_path):
    """A real SQLite file (not in-memory) so read-only mode can be tested."""
    path = tmp_path / "test.db"
    eng = create_engine(f"sqlite:///{path}")
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE events (id INTEGER PRIMARY KEY, label TEXT)"))
        conn.execute(text("INSERT INTO events VALUES (1, 'alpha'), (2, 'beta')"))
    eng.dispose()
    return path


@pytest.fixture()
def ro_engine(db_path):
    return make_readonly_engine(f"sqlite:///{db_path}")


# ── read access still works ────────────────────────────────────────────────────

def test_readonly_engine_allows_select(ro_engine):
    with ro_engine.connect() as conn:
        rows = conn.execute(text("SELECT * FROM events")).fetchall()

    assert len(rows) == 2


def test_readonly_engine_allows_select_with_where(ro_engine):
    with ro_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT label FROM events WHERE id = 1")
        ).fetchall()

    assert rows[0][0] == "alpha"


# ── write operations are rejected ─────────────────────────────────────────────

def test_readonly_engine_rejects_drop_table(ro_engine, db_path):
    with pytest.raises(Exception):
        with ro_engine.connect() as conn:
            conn.execute(text("DROP TABLE events"))

    # Table must still exist in the original file
    verify = create_engine(f"sqlite:///{db_path}")
    assert "events" in inspect(verify).get_table_names()


def test_readonly_engine_rejects_delete(ro_engine, db_path):
    with pytest.raises(Exception):
        with ro_engine.connect() as conn:
            conn.execute(text("DELETE FROM events"))

    verify = create_engine(f"sqlite:///{db_path}")
    with verify.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM events")).scalar()
    assert count == 2


def test_readonly_engine_rejects_insert(ro_engine, db_path):
    with pytest.raises(Exception):
        with ro_engine.connect() as conn:
            conn.execute(text("INSERT INTO events VALUES (3, 'gamma')"))

    verify = create_engine(f"sqlite:///{db_path}")
    with verify.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM events")).scalar()
    assert count == 2


def test_readonly_engine_rejects_create_table(ro_engine):
    with pytest.raises(Exception):
        with ro_engine.connect() as conn:
            conn.execute(text("CREATE TABLE new_table (x INTEGER)"))
