"""
Tests for _probe_connection — early database reachability check.

For non-SQLite databases, the connection is probed before any introspection or
LLM calls. This prevents wasting tokens on an unreachable database.

SQLite has its own file-existence check so it is excluded from the probe.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine

from schematica.agent import _probe_connection


# ── reachable database passes silently ────────────────────────────────────────

def test_probe_passes_silently_for_reachable_sqlite_engine():
    engine = create_engine("sqlite:///:memory:")

    # Must not raise
    _probe_connection(engine, "sqlite:///:memory:")


# ── unreachable database raises SystemExit ────────────────────────────────────

def test_probe_raises_system_exit_when_connect_raises():
    engine = MagicMock()
    engine.connect.side_effect = Exception("connection refused")

    with pytest.raises(SystemExit):
        _probe_connection(engine, "postgresql://host/db")


def test_probe_system_exit_message_includes_connection_string():
    engine = MagicMock()
    engine.connect.side_effect = Exception("timeout")

    with pytest.raises(SystemExit):
        _probe_connection(engine, "postgresql://host/db")


# ── SQLite is exempt from probe ───────────────────────────────────────────────

def test_probe_skips_connect_call_for_sqlite():
    # SQLite file existence is checked separately; the probe must not double-check.
    engine = MagicMock()

    _probe_connection(engine, "sqlite:///path/to/file.db")

    engine.connect.assert_not_called()
