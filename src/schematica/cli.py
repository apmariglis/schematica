"""
cli.py — Command-line entry point and output path utilities.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from urllib.parse import urlparse

from schematica import agent
from schematica.db import prompt_readonly_confirmation

_KNOWN_SCHEMES = ("sqlite:///", "postgresql://", "mysql://", "mssql://", "oracle://")


def _to_connection_string(db: str) -> str:
    """Convert a file path to a SQLite connection string; pass through existing connection strings."""
    if any(db.startswith(s) for s in _KNOWN_SCHEMES):
        return db
    return f"sqlite:///{db}"


def _model_folder_name(model: str) -> str:
    """
    Return a filesystem-safe folder name derived from MODEL.

    'gemini/gemini-2.5-flash'              →  'gemini-2.5-flash'
    'anthropic/claude-haiku-4-5-20251001'  →  'claude-haiku-4-5-20251001'
    """
    name = model.split("/", 1)[-1] if "/" in model else model
    return re.sub(r"[^\w.\-]", "_", name)


def _next_catalogue_index(out_dir: Path, db_name: str) -> int:
    """Return the next available 1-based index for <db_name>_catalogue_<n>.json."""
    if not out_dir.exists():
        return 1
    existing = list(out_dir.glob(f"{db_name}_catalogue_*.json"))
    indices = []
    for p in existing:
        stem = p.stem  # e.g. solar_wind_catalogue_3
        suffix = stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            indices.append(int(suffix))
    return max(indices, default=0) + 1


def _derive_catalogue_path(connection_string: str, model: str) -> str:
    """
    Derive the catalogue output path from a connection string.

    Catalogues are stored in data/<model>/<db_name>_catalogue_<n>.json,
    where <n> auto-increments so repeated runs never overwrite each other.

    sqlite:///data/solar_wind.db  →  data/gemini-2.5-flash/solar_wind_catalogue_1.json
    postgresql://.../mydb         →  data/gemini-2.5-flash/mydb_catalogue_1.json
    """
    if connection_string.startswith("sqlite:///"):
        db_file = Path(connection_string[len("sqlite:///") :])
        db_name = db_file.stem
        data_dir = db_file.parent
    else:
        parsed = urlparse(connection_string)
        db_name = parsed.path.lstrip("/")
        data_dir = Path("data")

    out_dir = data_dir / _model_folder_name(model)
    idx = _next_catalogue_index(out_dir, db_name)
    return str(out_dir / f"{db_name}_catalogue_{idx}.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Schematica — analyse a database and produce a data catalogue.",
    )
    parser.add_argument(
        "--db",
        required=True,
        metavar="DB",
        help="Database file path (e.g. ./data/mydb.db) or SQLAlchemy connection string",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="OUTPUT_JSON",
        help="Path to write the catalogue JSON (default: auto-derived from db name)",
    )
    parser.add_argument(
        "--skip-ro-check",
        action="store_true",
        help="Skip the read-only user confirmation prompt (for CI / automated use)",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Override SC_MODEL from .env (e.g. gpt-4o, gemini/gemini-2.5-flash)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        default=False,
        help="Enable prompt caching (anthropic/ models only). Overrides SC_CACHE=false.",
    )
    return parser.parse_args()


def _assign_model(args: argparse.Namespace) -> None:
    if args.model:
        is_anthropic = args.model.startswith(agent._ANTHROPIC_PREFIX)
        if is_anthropic:
            if args.cache:
                agent._apply_model_override(args.model, cache_override=True)
            else:
                agent._apply_model_override(args.model, cache_override=False)
        else:
            if args.cache:
                warnings.warn(
                    f"--cache has no effect with non-Anthropic model {args.model!r}; ignoring.",
                    UserWarning,
                    stacklevel=1,
                )
            agent._apply_model_override(args.model, cache_override=False)
    elif args.cache:
        agent._apply_model_override(agent._config.model, cache_override=True)


def main() -> None:
    """
    Console script entry point.

    Usage:
      schematica --db path/to/mydb.db
      schematica --db sqlite:///path/to/mydb.db
      schematica --db postgresql://user:pass@host:5432/mydb
      schematica --db path/to/mydb.db --out path/to/custom.json
    """
    args = _parse_args()

    _assign_model(args)

    connection_string = _to_connection_string(args.db)
    prompt_readonly_confirmation(connection_string, skip=args.skip_ro_check)
    out_path = args.out or _derive_catalogue_path(
        connection_string, agent._config.model
    )
    print(f"Output → {out_path}", file=sys.stderr)

    try:
        agent.run(connection_string=connection_string, out_path=out_path)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
