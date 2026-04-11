"""
introspect.py — database schema introspection for the Schematica.

Uses SQLAlchemy so it works with any supported dialect:
  sqlite:///path/to/file.db
  postgresql://user:pass@host:port/dbname
  mysql+pymysql://user:pass@host:port/dbname
  mssql+pyodbc://user:pass@dsn

Returns a structured snapshot — tables, columns, types, FK relationships,
basic statistics, and sample rows. No business data beyond a handful of
sample values; the goal is giving the LLM enough context to reason about
what is measurable, not to transfer the data itself.
"""
from __future__ import annotations

import json
from urllib.parse import urlparse, urlunparse

from sqlalchemy import inspect, text

from schematica.db import make_engine


# Column name fragments that suggest a date/time value even when stored as TEXT
_DATE_NAME_HINTS = ("_dt", "dt_", "date", "time", "period", "month", "year")

# SQL type fragments that indicate numeric values
_NUMERIC_TYPE_HINTS = ("INT", "REAL", "FLOAT", "NUMERIC", "DECIMAL", "DOUBLE", "MONEY", "NUMBER")

# SQL type fragments that indicate date/time values
_DATE_TYPE_HINTS = ("DATE", "TIME", "TIMESTAMP", "DATETIME")


def introspect(connection_string: str) -> dict:
    """
    Return a comprehensive schema snapshot of any SQLAlchemy-supported database.

    Structure:
    {
      "connection_string": "<redacted>",
      "dialect": "sqlite" | "postgresql" | ...,
      "tables": [
        {
          "name": str,
          "row_count": int,
          "columns": [
            {
              "name": str,
              "type": str,
              "nullable": bool,
              "primary_key": bool,
              "stats": {
                # numeric / date:  min, max, n_null
                # text:            n_distinct, top_values (dict), n_null
              }
            }
          ],
          "foreign_keys": [{"from_col", "to_table", "to_col"}],
          "sample_rows": [...]   # up to 5 rows as dicts
        }
      ]
    }
    """
    engine = make_engine(connection_string)
    insp   = inspect(engine)
    dialect = engine.dialect.name

    tables = []
    for table_name in insp.get_table_names():
        tables.append(_introspect_table(engine, insp, table_name))

    return {
        "connection_string": _redact(connection_string),
        "dialect": dialect,
        "tables": tables,
    }


def render_as_text(snapshot: dict) -> str:
    """Compact human-readable rendering used as context in the agent's initial message."""
    lines = [
        f"DATABASE: {snapshot['connection_string']}",
        f"DIALECT:  {snapshot['dialect']}",
        "",
    ]
    for t in snapshot["tables"]:
        lines.append(f"TABLE: {t['name']}  ({t['row_count']:,} rows)")
        for c in t["columns"]:
            s = c["stats"]
            pk  = " [PK]" if c["primary_key"] else ""
            req = "" if c["nullable"] else " NOT NULL"
            if s.get("binary"):
                detail = "binary/blob — skipped"
            elif "min" in s:
                detail = f"range=[{s['min']} → {s['max']}]  nulls={s['n_null']}"
            else:
                top = list(s.get("top_values", {}).items())[:5]
                top_str = ", ".join(f"{v}({n})" for v, n in top)
                detail = f"distinct={s['n_distinct']}  top=[{top_str}]  nulls={s['n_null']}"
            lines.append(f"  {c['name']}{pk}  {c['type']}{req}  — {detail}")
        for fk in t["foreign_keys"]:
            lines.append(f"  FK: {fk['from_col']} → {fk['to_table']}.{fk['to_col']}")
        if t["sample_rows"]:
            lines.append(f"  SAMPLE ROW: {json.dumps(t['sample_rows'][0], default=str)}")
        lines.append("")
    return "\n".join(lines)


# ── private ────────────────────────────────────────────────────────────────────

def _redact(connection_string: str) -> str:
    try:
        p = urlparse(connection_string)
        if not p.password:
            return connection_string
        netloc = p.hostname or ""
        if p.username:
            netloc = f"{p.username}:***@{netloc}"
        if p.port:
            netloc = f"{netloc}:{p.port}"
        p = p._replace(netloc=netloc)
        return urlunparse(p)
    except Exception:
        return connection_string


def _introspect_table(engine, insp, table_name: str) -> dict:
    pk_cols = set(insp.get_pk_constraint(table_name).get("constrained_columns", []))
    fk_list = [
        {
            "from_col": fk["constrained_columns"][0],
            "to_table": fk["referred_table"],
            "to_col":   fk["referred_columns"][0],
        }
        for fk in insp.get_foreign_keys(table_name)
        if fk["constrained_columns"] and fk["referred_columns"]
    ]

    with engine.connect() as conn:
        row_count = conn.execute(
            text(f'SELECT COUNT(*) FROM "{table_name}"')
        ).scalar()

        columns = [
            _introspect_column(conn, table_name, col, col["name"] in pk_cols)
            for col in insp.get_columns(table_name)
        ]
        sample_rows = _sample_rows(conn, table_name)

    return {
        "name": table_name,
        "row_count": row_count,
        "columns": columns,
        "foreign_keys": fk_list,
        "sample_rows": sample_rows,
    }


def _introspect_column(conn, table: str, col_meta: dict, is_pk: bool) -> dict:
    name     = col_meta["name"]
    type_str = str(col_meta["type"]).upper()
    nullable = col_meta.get("nullable", True)

    is_blob    = "BLOB" in type_str
    is_numeric = any(h in type_str for h in _NUMERIC_TYPE_HINTS)
    is_date    = any(h in type_str for h in _DATE_TYPE_HINTS)

    # Text columns whose name looks date-related (common in SQLite where dates are TEXT)
    if not is_date and any(h in name.lower() for h in _DATE_NAME_HINTS):
        is_date = True

    # BLOB columns: skip statistics entirely — binary content is not useful as a metric
    if is_blob:
        return {
            "name":        name,
            "type":        str(col_meta["type"]),
            "nullable":    nullable,
            "primary_key": is_pk,
            "stats":       {"binary": True},
        }

    n_null = conn.execute(
        text(f'SELECT COUNT(*) FROM "{table}" WHERE "{name}" IS NULL')
    ).scalar()

    if is_numeric or is_date:
        row = conn.execute(
            text(f'SELECT MIN("{name}"), MAX("{name}") FROM "{table}" WHERE "{name}" IS NOT NULL')
        ).fetchone()
        stats = {"min": row[0], "max": row[1], "n_null": n_null}
    else:
        n_distinct = conn.execute(
            text(f'SELECT COUNT(DISTINCT "{name}") FROM "{table}"')
        ).scalar()
        top = conn.execute(
            text(
                f'SELECT "{name}", COUNT(*) AS cnt FROM "{table}" '
                f'WHERE "{name}" IS NOT NULL '
                f'GROUP BY "{name}" ORDER BY cnt DESC'
            )
        ).fetchmany(8)
        stats = {
            "n_distinct": n_distinct,
            "top_values": {str(r[0]): r[1] for r in top},
            "n_null": n_null,
        }

    return {
        "name":        name,
        "type":        str(col_meta["type"]),
        "nullable":    nullable,
        "primary_key": is_pk,
        "stats":       stats,
    }


def _sample_rows(conn, table: str, n: int = 5) -> list[dict]:
    rows = conn.execute(text(f'SELECT * FROM "{table}" LIMIT {n}')).mappings().fetchall()
    sanitised = []
    for row in rows:
        sanitised.append({
            k: "<binary>" if isinstance(v, (bytes, bytearray)) else v
            for k, v in dict(row).items()
        })
    return sanitised
