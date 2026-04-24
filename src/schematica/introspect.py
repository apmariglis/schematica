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

from schematica.db import make_readonly_engine


# Column name fragments that suggest a date/time value even when stored as TEXT
_DATE_NAME_HINTS = ("_dt", "dt_", "_at", "_ts", "date", "time", "period", "month", "year")

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
          "foreign_keys": [{"from_cols": [...], "to_table": str, "to_cols": [...]}],
          "sample_rows": [...]   # up to 5 rows as dicts
        }
      ]
    }
    """
    engine = make_readonly_engine(connection_string)
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


# Columns with more distinct values than this are too high-cardinality to be
# useful as metric breakdown dimensions (e.g. hundreds of free-text categories).
_DIMENSION_CARDINALITY_LIMIT = 20

# Reference/lookup tables are distinguished from fact tables by row count.
# A table with more rows than this is treated as a fact table whose FK
# column is not a useful dimension (e.g. account_id → accounts).
_MAX_LOOKUP_TABLE_ROWS = 200


def render_as_text(snapshot: dict) -> str:
    """Compact human-readable rendering used as context in the agent's initial message."""
    lines = [
        f"DATABASE: {snapshot['connection_string']}",
        f"DIALECT:  {snapshot['dialect']}",
        "",
    ]
    manifest = _build_dimension_manifest(snapshot)
    if manifest:
        lines.append(manifest)
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
            from_str = ", ".join(fk["from_cols"])
            to_str   = ", ".join(fk["to_cols"])
            lines.append(f"  FK: ({from_str}) → {fk['to_table']}.({to_str})")
        if t["sample_rows"]:
            lines.append(f"  SAMPLE ROW: {json.dumps(t['sample_rows'][0], default=str)}")
        lines.append("")
    return "\n".join(lines)


def _build_dimension_manifest(snapshot: dict) -> str:
    """Return a section listing every low-cardinality categorical column and its values.

    Included:
      - Text columns with n_distinct in [2, _DIMENSION_CARDINALITY_LIMIT]
      - FK columns whose target is a small lookup table with a text label column

    Excluded:
      - Primary keys (identifiers, not breakdown dimensions)
      - Numeric / date columns (they carry min/max stats, not top_values)
      - Unary columns (n_distinct == 1 — no breakdown is possible)
      - Columns above the cardinality limit
      - FK columns pointing to large fact tables

    The result is injected into the schema text passed to every LLM phase so the
    model always sees the complete set of dimension values without having to
    discover them through exploratory queries.
    """
    tables_by_name = {t["name"]: t for t in snapshot["tables"]}

    # (table_name, col_name) → target_table_name
    fk_targets: dict[tuple[str, str], str] = {}
    for t in snapshot["tables"]:
        for fk in t.get("foreign_keys", []):
            for from_col in fk["from_cols"]:
                fk_targets[(t["name"], from_col)] = fk["to_table"]

    def _label_values(table_name: str) -> list[str] | None:
        """Return sorted distinct values of the first suitable label column, or None."""
        t = tables_by_name.get(table_name)
        if not t or t.get("row_count", 0) > _MAX_LOOKUP_TABLE_ROWS:
            return None
        for col in t["columns"]:
            if col["primary_key"]:
                continue
            stats = col["stats"]
            n = stats.get("n_distinct", 0)
            if "top_values" in stats and 2 <= n <= _DIMENSION_CARDINALITY_LIMIT:
                return sorted(stats["top_values"].keys())
        return None

    lines = []
    for t in snapshot["tables"]:
        table_name = t["name"]
        for col in t["columns"]:
            if col["primary_key"]:
                continue
            col_name = col["name"]
            stats    = col["stats"]

            if "top_values" in stats:
                n = stats.get("n_distinct", 0)
                if 2 <= n <= _DIMENSION_CARDINALITY_LIMIT:
                    values = sorted(stats["top_values"].keys())
                    # Skip columns whose values look like JSON — they are structured
                    # payloads stored in text columns, not categorical dimensions.
                    if any(v.startswith("{") or v.startswith("[") for v in values):
                        continue
                    lines.append(f"  {table_name}.{col_name}  →  {', '.join(values)}")

            elif (table_name, col_name) in fk_targets:
                target = fk_targets[(table_name, col_name)]
                label_values = _label_values(target)
                if label_values:
                    lines.append(
                        f"  {table_name}.{col_name} (via {target})  →  {', '.join(label_values)}"
                    )

    if not lines:
        return ""

    return "DIMENSION BREAKDOWNS AVAILABLE\n" + "\n".join(lines) + "\n"


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
            "from_cols": fk["constrained_columns"],
            "to_table":  fk["referred_table"],
            "to_cols":   fk["referred_columns"],
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
