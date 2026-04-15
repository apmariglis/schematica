"""
Tests for _check_package_versions — startup guard against incompatible
pandas + numpy combinations.

Root cause: pandas < 2.2 calls numpy.rec internally during pd.read_sql().
numpy 2.0 removed numpy.rec. The combination silently corrupts every eval
result with "eval error: No module named 'numpy.rec'" after a full agent run.

The fix: fail immediately at startup with a clear error and fix command,
before wasting budget on a run whose Phase 3 eval will fail entirely.
"""
from __future__ import annotations

import pytest

from schematica.agent import _check_package_versions


# ── incompatible combination raises ───────────────────────────────────────────

def test_raises_when_old_pandas_with_new_numpy():
    with pytest.raises(RuntimeError):
        _check_package_versions(pandas_version="1.5.3", numpy_version="2.0.0")


def test_raises_when_pandas_2_1_with_numpy_2():
    # 2.1.x is still < 2.2 and therefore broken with numpy 2.x
    with pytest.raises(RuntimeError):
        _check_package_versions(pandas_version="2.1.4", numpy_version="2.0.0")


def test_raises_when_pandas_2_1_with_numpy_2_4():
    with pytest.raises(RuntimeError):
        _check_package_versions(pandas_version="2.1.4", numpy_version="2.4.4")


# ── error message is actionable ───────────────────────────────────────────────

def test_error_mentions_installed_pandas_version():
    with pytest.raises(RuntimeError, match="1.5.3"):
        _check_package_versions(pandas_version="1.5.3", numpy_version="2.0.0")


def test_error_mentions_installed_numpy_version():
    with pytest.raises(RuntimeError, match="2.0.0"):
        _check_package_versions(pandas_version="1.5.3", numpy_version="2.0.0")


def test_error_mentions_fix_command():
    with pytest.raises(RuntimeError, match="pandas>=2.2"):
        _check_package_versions(pandas_version="1.5.3", numpy_version="2.0.0")


# ── compatible combinations pass silently ─────────────────────────────────────

def test_passes_when_pandas_2_2_with_numpy_2():
    _check_package_versions(pandas_version="2.2.0", numpy_version="2.0.0")


def test_passes_when_pandas_3_with_numpy_2():
    _check_package_versions(pandas_version="3.0.2", numpy_version="2.4.4")


def test_passes_when_old_pandas_with_old_numpy():
    # numpy < 2.0 still has numpy.rec — no issue regardless of pandas version
    _check_package_versions(pandas_version="1.5.3", numpy_version="1.26.4")


def test_passes_when_pandas_2_2_with_old_numpy():
    _check_package_versions(pandas_version="2.2.0", numpy_version="1.26.4")
