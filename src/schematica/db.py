"""
db.py — SQLAlchemy engine factory for the Schematica.

Centralises engine creation so connection-level concerns (encoding quirks,
dialect workarounds, future SSL / pooling config) live in one place.
"""
from __future__ import annotations

import sqlite3

from sqlalchemy import event, create_engine
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

    Guarantees that no SQL can alter the database, regardless of what the agent
    produces:

    - SQLite: opens the file with mode=ro via the SQLite URI interface — the
      driver itself refuses writes at the OS level, including DDL that would
      normally auto-commit past a transaction boundary.
    - All other databases: registers a before_cursor_execute listener that raises
      PermissionError before any non-SELECT statement reaches the wire.
    """
    if connection_string.startswith("sqlite"):
        db_path = connection_string.split("///", 1)[-1]

        def _readonly_creator() -> sqlite3.Connection:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            return conn

        return create_engine("sqlite://", creator=_readonly_creator)

    engine = create_engine(connection_string)

    @event.listens_for(engine, "before_cursor_execute")
    def _reject_writes(conn, cursor, statement, parameters, context, executemany):
        first_token = statement.strip().split()[0].upper() if statement.strip() else ""
        if first_token not in ("SELECT", "WITH", "EXPLAIN"):
            raise PermissionError(
                f"Write operations are not permitted during exploration: {first_token}"
            )

    return engine
