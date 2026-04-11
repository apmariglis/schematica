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
