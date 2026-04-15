"""
Tests that the Phase 1 catalogue is written even when Phase 3 crashes.

Before this fix, _write_output was only called after _run_phase3 returned
successfully. Any exception in Phase 3 (network error, [Errno 2], numpy crash,
etc.) would bubble up through run() and the catalogue file would never be
written, losing all Phase 1 work.

The fix: wrap _run_phase3 in a try/except inside run() so _write_output is
always reached, using the Phase 1 catalogue as a fallback.

Tested via _run_phase3_safe — the extracted wrapper that implements this logic.
"""
from __future__ import annotations

import pytest

from schematica.agent import _run_phase3_safe


class _FakeCatalogue:
    """Minimal stand-in for DataCatalogue."""
    name = "phase1"


class _FakeEngine:
    pass


def _noop_phase3(*args, **kwargs):
    refined = _FakeCatalogue()
    refined.name = "refined"
    return refined, [], [], []


def _crashing_phase3(*args, **kwargs):
    raise FileNotFoundError("[Errno 2] No such file or directory")


def _network_crashing_phase3(*args, **kwargs):
    raise RuntimeError("Connection refused")


# ── successful phase 3 returns refined catalogue ──────────────────────────────

def test_returns_refined_catalogue_when_phase3_succeeds():
    phase1 = _FakeCatalogue()

    result, _, _, _ = _run_phase3_safe(_noop_phase3, phase1, "schema", _FakeEngine(), {}, {})

    assert result.name == "refined"


# ── phase 3 crash falls back to phase 1 catalogue ─────────────────────────────

def test_returns_phase1_catalogue_when_phase3_raises_file_error():
    phase1 = _FakeCatalogue()

    result, _, _, _ = _run_phase3_safe(_crashing_phase3, phase1, "schema", _FakeEngine(), {}, {})

    assert result.name == "phase1"


def test_returns_phase1_catalogue_when_phase3_raises_runtime_error():
    phase1 = _FakeCatalogue()

    result, _, _, _ = _run_phase3_safe(_network_crashing_phase3, phase1, "schema", _FakeEngine(), {}, {})

    assert result.name == "phase1"


def test_returns_empty_metric_results_on_phase3_crash():
    phase1 = _FakeCatalogue()

    _, metric_results, _, _ = _run_phase3_safe(_crashing_phase3, phase1, "schema", _FakeEngine(), {}, {})

    assert metric_results == []


def test_returns_empty_fact_results_on_phase3_crash():
    phase1 = _FakeCatalogue()

    _, _, fact_results, _ = _run_phase3_safe(_crashing_phase3, phase1, "schema", _FakeEngine(), {}, {})

    assert fact_results == []


def test_returns_empty_uncovered_tables_on_phase3_crash():
    phase1 = _FakeCatalogue()

    _, _, _, uncovered = _run_phase3_safe(_crashing_phase3, phase1, "schema", _FakeEngine(), {}, {})

    assert uncovered == []
