"""
Tests for schematica.pricing — specifically the cache write failure behaviour.

_save_cache must warn (not silently pass) when it cannot write to disk, so
users in restricted environments know why pricing is never cached.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from schematica.pricing import _save_cache


def test_save_cache_writes_file(tmp_path):
    cache_path = tmp_path / "sub" / "pricing.json"
    pricing = {"gpt-4": {"input": 10.0, "output": 30.0}}

    _save_cache(pricing, cache_path)

    assert cache_path.exists()
    assert json.loads(cache_path.read_text()) == pricing


def test_save_cache_emits_warning_on_permission_error(tmp_path):
    # Make the directory read-only so the file write fails.
    cache_dir = tmp_path / "ro"
    cache_dir.mkdir()
    cache_dir.chmod(0o444)
    cache_path = cache_dir / "pricing.json"

    with pytest.warns(UserWarning, match="pricing cache"):
        _save_cache({"model": {"input": 1.0, "output": 1.0}}, cache_path)

    # Cleanup: restore permissions so tmp_path teardown can delete the dir.
    cache_dir.chmod(0o755)
