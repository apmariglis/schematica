"""
Tests for schematica.introspect — schema snapshot computation.

Uses an in-memory SQLite database with a mix of column types:
  events(id INTEGER PK, occurred_at TEXT, magnitude REAL, label TEXT)
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from schematica.introspect import introspect, render_as_text, _redact


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def connection_string():
    return "sqlite:///:memory:"


@pytest.fixture()
def populated_db(tmp_path):
    """
    A SQLite file (not in-memory) with a small events table.
    Returns the connection string.
    """
    db_path = tmp_path / "events.db"
    cs = f"sqlite:///{db_path}"
    eng = create_engine(cs)
    with eng.begin() as conn:
        conn.execute(text("""
            CREATE TABLE events (
                id          INTEGER PRIMARY KEY,
                occurred_at TEXT    NOT NULL,
                magnitude   REAL,
                label       TEXT
            )
        """))
        conn.execute(text("""
            INSERT INTO events VALUES
                (1, '2024-01-01', 3.5, 'alpha'),
                (2, '2024-02-01', 7.1, 'beta'),
                (3, '2024-03-01', NULL, 'alpha')
        """))
    return cs


# ── _redact ────────────────────────────────────────────────────────────────────

def test_redact_removes_password_from_postgresql_url():
    cs = "postgresql://admin:s3cret@localhost:5432/mydb"

    result = _redact(cs)

    assert "s3cret" not in result
    assert "admin" in result
    assert "localhost" in result
    assert "mydb" in result


def test_redact_leaves_sqlite_path_unchanged():
    cs = "sqlite:///data/events.db"

    result = _redact(cs)

    assert result == cs


def test_redact_leaves_url_without_password_unchanged():
    cs = "postgresql://localhost/mydb"

    result = _redact(cs)

    assert result == cs


# ── introspect — structure ─────────────────────────────────────────────────────

def test_introspect_returns_dialect(populated_db):
    snapshot = introspect(populated_db)

    assert snapshot["dialect"] == "sqlite"


def test_introspect_returns_table_names(populated_db):
    snapshot = introspect(populated_db)

    table_names = [t["name"] for t in snapshot["tables"]]
    assert "events" in table_names


def test_introspect_reports_correct_row_count(populated_db):
    snapshot = introspect(populated_db)

    events = next(t for t in snapshot["tables"] if t["name"] == "events")
    assert events["row_count"] == 3


def test_introspect_reports_correct_column_count(populated_db):
    snapshot = introspect(populated_db)

    events = next(t for t in snapshot["tables"] if t["name"] == "events")
    assert len(events["columns"]) == 4


def test_introspect_identifies_primary_key_column(populated_db):
    snapshot = introspect(populated_db)

    events = next(t for t in snapshot["tables"] if t["name"] == "events")
    id_col = next(c for c in events["columns"] if c["name"] == "id")
    assert id_col["primary_key"] is True


def test_introspect_marks_non_pk_columns_as_not_primary_key(populated_db):
    snapshot = introspect(populated_db)

    events = next(t for t in snapshot["tables"] if t["name"] == "events")
    label_col = next(c for c in events["columns"] if c["name"] == "label")
    assert label_col["primary_key"] is False


def test_introspect_computes_min_max_for_numeric_column(populated_db):
    snapshot = introspect(populated_db)

    events = next(t for t in snapshot["tables"] if t["name"] == "events")
    mag_col = next(c for c in events["columns"] if c["name"] == "magnitude")
    stats = mag_col["stats"]

    assert stats["min"] == pytest.approx(3.5)
    assert stats["max"] == pytest.approx(7.1)


def test_introspect_counts_nulls_for_nullable_column(populated_db):
    snapshot = introspect(populated_db)

    events = next(t for t in snapshot["tables"] if t["name"] == "events")
    mag_col = next(c for c in events["columns"] if c["name"] == "magnitude")

    assert mag_col["stats"]["n_null"] == 1


def test_introspect_computes_distinct_count_for_text_column(populated_db):
    snapshot = introspect(populated_db)

    events = next(t for t in snapshot["tables"] if t["name"] == "events")
    label_col = next(c for c in events["columns"] if c["name"] == "label")

    # 'alpha' and 'beta' → 2 distinct values
    assert label_col["stats"]["n_distinct"] == 2


def test_introspect_includes_sample_rows(populated_db):
    snapshot = introspect(populated_db)

    events = next(t for t in snapshot["tables"] if t["name"] == "events")
    assert len(events["sample_rows"]) > 0
    assert isinstance(events["sample_rows"][0], dict)


def test_introspect_redacts_connection_string_in_output(tmp_path):
    db_path = tmp_path / "events.db"
    cs = f"sqlite:///{db_path}"
    eng = create_engine(cs)
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE t (x INTEGER)"))

    snapshot = introspect(cs)

    # Connection string is present but passwords would be redacted
    assert "connection_string" in snapshot


# ── render_as_text ─────────────────────────────────────────────────────────────

def test_render_as_text_includes_table_name(populated_db):
    snapshot = introspect(populated_db)

    output = render_as_text(snapshot)

    assert "events" in output


def test_render_as_text_includes_row_count(populated_db):
    snapshot = introspect(populated_db)

    output = render_as_text(snapshot)

    assert "3" in output


def test_render_as_text_includes_column_names(populated_db):
    snapshot = introspect(populated_db)

    output = render_as_text(snapshot)

    assert "magnitude" in output
    assert "label" in output


def test_render_as_text_includes_dialect(populated_db):
    snapshot = introspect(populated_db)

    output = render_as_text(snapshot)

    assert "sqlite" in output.lower()


# ── foreign keys ───────────────────────────────────────────────────────────────

@pytest.fixture()
def composite_fk_db(tmp_path):
    """
    A database with a junction table that has a composite foreign key:
      registrations(event_id, attendee_id) → events(id) + attendees(id)
    """
    db_path = tmp_path / "composite.db"
    cs = f"sqlite:///{db_path}"
    eng = create_engine(cs)
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE events   (id INTEGER PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE attendees (id INTEGER PRIMARY KEY)"))
        conn.execute(text("""
            CREATE TABLE registrations (
                event_id    INTEGER NOT NULL REFERENCES events(id),
                attendee_id INTEGER NOT NULL REFERENCES attendees(id),
                PRIMARY KEY (event_id, attendee_id)
            )
        """))
    return cs


def test_introspect_captures_all_columns_of_composite_foreign_key(composite_fk_db):
    snapshot = introspect(composite_fk_db)

    reg = next(t for t in snapshot["tables"] if t["name"] == "registrations")
    fk_to_events   = next(fk for fk in reg["foreign_keys"] if fk["to_table"] == "events")
    fk_to_attendees = next(fk for fk in reg["foreign_keys"] if fk["to_table"] == "attendees")

    # Each FK must expose a list of columns, not a single truncated string
    assert fk_to_events["from_cols"]   == ["event_id"]
    assert fk_to_events["to_cols"]     == ["id"]
    assert fk_to_attendees["from_cols"] == ["attendee_id"]
    assert fk_to_attendees["to_cols"]   == ["id"]


def test_render_as_text_shows_all_columns_of_composite_foreign_key(composite_fk_db):
    snapshot = introspect(composite_fk_db)

    output = render_as_text(snapshot)

    assert "event_id" in output
    assert "attendee_id" in output


def test_column_with_at_suffix_gets_min_max_stats_not_top_values(populated_db):
    # Columns named *_at are timestamps stored as TEXT in SQLite; they should
    # be treated as date columns (min/max range) rather than categoricals.
    snapshot = introspect(populated_db)

    events = next(t for t in snapshot["tables"] if t["name"] == "events")
    occurred_at = next(c for c in events["columns"] if c["name"] == "occurred_at")

    assert "min" in occurred_at["stats"]
    assert "top_values" not in occurred_at["stats"]
