"""
Tests for prompt_readonly_confirmation — the read-only safety check shown
before connecting to a non-SQLite database.
"""
from __future__ import annotations

import pytest

from schematica.db import prompt_readonly_confirmation


# ── SQLite: always silent ──────────────────────────────────────────────────────

def test_sqlite_connection_string_skips_prompt():
    # Should return silently — no input needed, no SystemExit raised.
    prompt_readonly_confirmation("sqlite:///data/events.db", skip=False)


def test_sqlite_file_path_also_skips_prompt():
    prompt_readonly_confirmation("sqlite:///data/events.db", skip=False)


# ── non-SQLite + skip=True: always silent ─────────────────────────────────────

def test_skip_flag_suppresses_prompt_for_postgresql():
    prompt_readonly_confirmation("postgresql://user:pw@host/db", skip=True)


def test_skip_flag_suppresses_prompt_for_mysql():
    prompt_readonly_confirmation("mysql://user:pw@host/db", skip=True)


# ── non-SQLite + skip=False: prompts user ─────────────────────────────────────

def test_confirmed_yes_does_not_abort(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")

    prompt_readonly_confirmation("postgresql://user:pw@host/db", skip=False)


def test_confirmed_uppercase_yes_does_not_abort(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Y")

    prompt_readonly_confirmation("postgresql://user:pw@host/db", skip=False)


def test_declined_aborts_with_systemexit(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")

    with pytest.raises(SystemExit):
        prompt_readonly_confirmation("postgresql://user:pw@host/db", skip=False)


def test_empty_answer_aborts_with_systemexit(monkeypatch):
    # Default is N — pressing Enter without typing should abort.
    monkeypatch.setattr("builtins.input", lambda _: "")

    with pytest.raises(SystemExit):
        prompt_readonly_confirmation("postgresql://user:pw@host/db", skip=False)
