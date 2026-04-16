"""
Tests for _derive_catalogue_path — consistent output location for all DB types.

The function must produce the same path structure regardless of whether the
source is a SQLite file, a PostgreSQL connection string, or anything else.
Output is always: <out_dir>/<model_folder>/<db_stem>_catalogue_<n>.json
"""
from __future__ import annotations

import pytest

from schematica.cli import _derive_catalogue_path


# ── path structure ─────────────────────────────────────────────────────────────

def test_sqlite_path_uses_out_dir(tmp_path):
    result = _derive_catalogue_path("sqlite:///data/events.db", "gemini/gemini-2.5-flash", str(tmp_path))

    assert result.startswith(str(tmp_path))


def test_postgresql_uses_out_dir(tmp_path):
    result = _derive_catalogue_path("postgresql://user:pw@host/mydb", "gemini/gemini-2.5-flash", str(tmp_path))

    assert result.startswith(str(tmp_path))


def test_model_folder_placed_under_out_dir(tmp_path):
    result = _derive_catalogue_path("sqlite:///data/events.db", "gemini/gemini-2.5-flash", str(tmp_path))

    assert "gemini-2.5-flash" in result


def test_sqlite_stem_extracted_as_filename(tmp_path):
    result = _derive_catalogue_path("sqlite:///data/events.db", "gemini/gemini-2.5-flash", str(tmp_path))

    assert "events_catalogue_" in result


def test_postgresql_stem_extracted_as_dbname(tmp_path):
    result = _derive_catalogue_path("postgresql://user:pw@host/mydb", "gemini/gemini-2.5-flash", str(tmp_path))

    assert "mydb_catalogue_" in result


def test_output_file_has_json_extension(tmp_path):
    result = _derive_catalogue_path("sqlite:///data/events.db", "gemini/gemini-2.5-flash", str(tmp_path))

    assert result.endswith(".json")


# ── index auto-increment ───────────────────────────────────────────────────────

def test_first_run_gets_index_one(tmp_path):
    result = _derive_catalogue_path("sqlite:///data/events.db", "gemini/gemini-2.5-flash", str(tmp_path))

    assert result.endswith("_catalogue_1.json")


def test_second_run_gets_index_two(tmp_path):
    first = _derive_catalogue_path("sqlite:///data/events.db", "gemini/gemini-2.5-flash", str(tmp_path))
    from pathlib import Path
    Path(first).parent.mkdir(parents=True, exist_ok=True)
    Path(first).touch()

    second = _derive_catalogue_path("sqlite:///data/events.db", "gemini/gemini-2.5-flash", str(tmp_path))

    assert second.endswith("_catalogue_2.json")


# ── same structure for all db types ──────────────────────────────────────────

def test_sqlite_and_postgresql_produce_same_structure(tmp_path):
    sqlite_dir  = tmp_path / "sqlite"
    pg_dir      = tmp_path / "pg"
    model       = "gemini/gemini-2.5-flash"

    sqlite_result = _derive_catalogue_path("sqlite:///data/mydb.db", model, str(sqlite_dir))
    pg_result     = _derive_catalogue_path("postgresql://user:pw@host/mydb", model, str(pg_dir))

    # Both should have the same filename structure
    from pathlib import Path
    assert Path(sqlite_result).name == Path(pg_result).name
