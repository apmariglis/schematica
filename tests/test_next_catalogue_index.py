"""
Tests for _next_catalogue_index — handles non-existent output directories.

On Python 3.11, Path.glob() raises FileNotFoundError when the directory does
not exist. Python 3.12+ returns an empty iterator. The function must work on
both by returning index 1 when the directory is absent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from schematica.agent import _next_catalogue_index


def test_returns_one_when_directory_does_not_exist(tmp_path):
    missing_dir = tmp_path / "nonexistent" / "subdir"

    result = _next_catalogue_index(missing_dir, "solar_wind")

    assert result == 1


def test_returns_one_when_directory_is_empty(tmp_path):
    empty_dir = tmp_path / "output"
    empty_dir.mkdir()

    result = _next_catalogue_index(empty_dir, "solar_wind")

    assert result == 1


def test_returns_two_when_one_catalogue_exists(tmp_path):
    out_dir = tmp_path / "model"
    out_dir.mkdir()
    (out_dir / "solar_wind_catalogue_1.json").touch()

    result = _next_catalogue_index(out_dir, "solar_wind")

    assert result == 2


def test_returns_next_after_gap_in_sequence(tmp_path):
    out_dir = tmp_path / "model"
    out_dir.mkdir()
    (out_dir / "solar_wind_catalogue_1.json").touch()
    (out_dir / "solar_wind_catalogue_3.json").touch()

    result = _next_catalogue_index(out_dir, "solar_wind")

    assert result == 4


def test_ignores_catalogues_for_different_db(tmp_path):
    out_dir = tmp_path / "model"
    out_dir.mkdir()
    (out_dir / "other_db_catalogue_5.json").touch()

    result = _next_catalogue_index(out_dir, "solar_wind")

    assert result == 1
