"""
db.py — SQLAlchemy engine factory for the Schematica.

Centralises engine creation so connection-level concerns (encoding quirks,
dialect workarounds, future SSL / pooling config) live in one place.
"""
from __future__ import annotations

import sqlite3

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def make_engine(connection_string: str) -> Engine:
    """
    Return a SQLAlchemy Engine for the given connection string.

    SQLite-specific: overrides text_factory so non-UTF-8 bytes (e.g.
    Windows-1252 data from legacy Access / SQL Server exports) are decoded
    with replacement characters instead of raising UnicodeDecodeError.
    """
    if connection_string.startswith("sqlite"):
        db_path = connection_string.split("///", 1)[-1]

        def _creator() -> sqlite3.Connection:
            conn = sqlite3.connect(db_path)
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            return conn

        return create_engine("sqlite://", creator=_creator)

    return create_engine(connection_string)


def make_readonly_engine(connection_string: str) -> Engine:
    """
    Return a read-only Engine for use during exploration queries.

    - SQLite: opens the file with mode=ro via the SQLite URI interface — the
      driver itself refuses writes at the OS level, including DDL that would
      normally auto-commit past a transaction boundary.
    - All other databases: returns a standard engine. SQLAlchemy runs its own
      internal session-management statements (SET, SHOW, PRAGMA) during
      connection setup and introspection; a blanket first-token listener would
      block those and break schema discovery. For PostgreSQL, MySQL, and other
      dialects the read-only guarantee must be enforced at the database level
      by connecting with a user that has only SELECT privileges.
    """
    if connection_string.startswith("sqlite"):
        db_path = connection_string.split("///", 1)[-1]

        def _readonly_creator() -> sqlite3.Connection:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            return conn

        return create_engine("sqlite://", creator=_readonly_creator)

    return create_engine(connection_string)
