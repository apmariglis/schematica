"""
Tests for _derive_catalogue_pattern — output path pattern for all DB types.

The function returns a pattern (no index, no extension) that _write_output
turns into an auto-indexed file.  Structure is always:
    <out_dir>/<model_folder>/<db_stem>_catalogue

Atomic index assignment and concurrent-run safety are tested via _write_output.
"""
from __future__ import annotations

import json
import os
import threading

import pytest

from schematica.cli import _derive_catalogue_pattern
from schematica.agent import _write_output
from schematica.catalogue import DataCatalogue


# ── helpers ────────────────────────────────────────────────────────────────────

def _minimal_catalogue() -> DataCatalogue:
    return DataCatalogue(
        analysed_at="2024-01-01T00:00:00",
        connection="sqlite:///test.db",
        dialect="sqlite",
        description="Test DB",
        overview="A test database.",
        tables=[],
        measurable_metrics=[],
        queryable_facts=[],
        time_coverage={"start": "2024-01-01", "end": "2024-12-31"},
        data_quality_notes=[],
        key_terms=[],
        table_relationships=[],
        model="test-model",
    )


# ── pattern structure ──────────────────────────────────────────────────────────

def test_sqlite_pattern_uses_out_dir(tmp_path):
    result = _derive_catalogue_pattern("sqlite:///data/events.db", "gemini/gemini-2.5-flash", str(tmp_path))

    assert result.startswith(str(tmp_path))


def test_postgresql_pattern_uses_out_dir(tmp_path):
    result = _derive_catalogue_pattern("postgresql://user:pw@host/mydb", "gemini/gemini-2.5-flash", str(tmp_path))

    assert result.startswith(str(tmp_path))


def test_model_folder_placed_under_out_dir(tmp_path):
    result = _derive_catalogue_pattern("sqlite:///data/events.db", "gemini/gemini-2.5-flash", str(tmp_path))

    assert "gemini-2.5-flash" in result


def test_sqlite_stem_extracted_as_filename(tmp_path):
    result = _derive_catalogue_pattern("sqlite:///data/events.db", "gemini/gemini-2.5-flash", str(tmp_path))

    assert "events_catalogue" in result


def test_postgresql_stem_extracted_as_dbname(tmp_path):
    result = _derive_catalogue_pattern("postgresql://user:pw@host/mydb", "gemini/gemini-2.5-flash", str(tmp_path))

    assert "mydb_catalogue" in result


def test_pattern_has_no_json_extension(tmp_path):
    # The pattern is passed to _write_output which appends _N.json atomically.
    result = _derive_catalogue_pattern("sqlite:///data/events.db", "gemini/gemini-2.5-flash", str(tmp_path))

    assert not result.endswith(".json")


def test_sqlite_and_postgresql_produce_same_stem_structure(tmp_path):
    model = "gemini/gemini-2.5-flash"

    sqlite_result = _derive_catalogue_pattern("sqlite:///data/mydb.db", model, str(tmp_path))
    pg_result = _derive_catalogue_pattern("postgresql://user:pw@host/mydb", model, str(tmp_path))

    # Both have the same base filename pattern (mydb_catalogue)
    from pathlib import Path
    assert Path(sqlite_result).name == Path(pg_result).name


# ── atomic index assignment via _write_output ─────────────────────────────────

def test_first_write_gets_index_one(tmp_path):
    pattern = str(tmp_path / "gemini-2.5-flash" / "events_catalogue")
    catalogue = _minimal_catalogue()

    _write_output(catalogue, pattern)

    assert (tmp_path / "gemini-2.5-flash" / "events_catalogue_1.json").exists()


def test_second_write_gets_index_two(tmp_path):
    pattern = str(tmp_path / "gemini-2.5-flash" / "events_catalogue")
    catalogue = _minimal_catalogue()

    _write_output(catalogue, pattern)
    _write_output(catalogue, pattern)

    assert (tmp_path / "gemini-2.5-flash" / "events_catalogue_2.json").exists()


def test_output_file_contains_valid_json(tmp_path):
    pattern = str(tmp_path / "model" / "db_catalogue")
    catalogue = _minimal_catalogue()

    _write_output(catalogue, pattern)

    written = list((tmp_path / "model").glob("db_catalogue_*.json"))
    assert len(written) == 1
    data = json.loads(written[0].read_text())
    assert "description" in data


def test_concurrent_writes_produce_unique_files(tmp_path):
    # Ten concurrent writers must each get a distinct index — no collisions.
    pattern = str(tmp_path / "model" / "db_catalogue")
    catalogue = _minimal_catalogue()
    errors = []

    def write():
        try:
            _write_output(catalogue, pattern)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    written = list((tmp_path / "model").glob("db_catalogue_*.json"))
    assert len(written) == 10
    assert len({f.name for f in written}) == 10  # all unique filenames
