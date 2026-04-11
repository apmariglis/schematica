"""
Tests for _snap_to_period — truncates a date string to the natural boundary
of a granularity. Covers: empty input, bare year, monthly, quarterly, annual,
and daily granularities, plus short/malformed strings.
"""
from __future__ import annotations

import pytest

from schematica.agent import _snap_to_period


# ── guard clauses ──────────────────────────────────────────────────────────────

def test_returns_empty_string_unchanged():
    assert _snap_to_period("", "monthly") == ""


def test_returns_string_shorter_than_seven_chars_unchanged():
    # e.g. "2024-0" — too short to be a valid YYYY-MM but not a bare year
    assert _snap_to_period("2024-0", "monthly") == "2024-0"


# ── bare year expansion ────────────────────────────────────────────────────────

def test_bare_year_expands_to_january_first():
    assert _snap_to_period("2024", "monthly") == "2024-01-01"


def test_bare_year_expansion_is_granularity_independent():
    # A bare year should always expand to YYYY-01-01 regardless of granularity
    assert _snap_to_period("1990", "annual") == "1990-01-01"
    assert _snap_to_period("2000", "daily")  == "2000-01-01"


# ── monthly granularity ────────────────────────────────────────────────────────

def test_monthly_granularity_snaps_to_first_of_month():
    assert _snap_to_period("2024-06-15", "monthly") == "2024-06-01"


def test_monthly_granularity_leaves_already_snapped_date_unchanged():
    assert _snap_to_period("2024-06-01", "monthly") == "2024-06-01"


# ── quarterly granularity ─────────────────────────────────────────────────────

def test_quarterly_granularity_snaps_to_first_of_month():
    # quarterly uses the same truncation as monthly (caller is responsible for
    # ensuring the month is a quarter start — snapping is just YYYY-MM-01)
    assert _snap_to_period("2024-09-30", "quarterly") == "2024-09-01"


# ── annual granularity ────────────────────────────────────────────────────────

def test_annual_granularity_snaps_to_january_first():
    assert _snap_to_period("2024-06-15", "annual") == "2024-01-01"


def test_annual_granularity_leaves_already_snapped_date_unchanged():
    assert _snap_to_period("2024-01-01", "annual") == "2024-01-01"


# ── unrecognised / daily granularity ─────────────────────────────────────────

def test_unrecognised_granularity_returns_date_unchanged():
    assert _snap_to_period("2024-06-15", "daily") == "2024-06-15"


def test_weekly_granularity_returns_date_unchanged():
    assert _snap_to_period("2024-06-15", "weekly") == "2024-06-15"
