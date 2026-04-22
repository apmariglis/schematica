"""
cli.py — Command-line entry point and output path utilities.
"""

from __future__ import annotations

import argparse
import os
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



def _db_stem(connection_string: str) -> str:
    """Extract a filesystem-safe database name from any connection string.

    sqlite:///data/solar_wind.db       →  'solar_wind'
    postgresql://user:pw@host/mydb     →  'mydb'
    """
    if connection_string.startswith("sqlite:///"):
        return Path(connection_string[len("sqlite:///"):].split("?")[0]).stem
    parsed = urlparse(connection_string)
    return parsed.path.lstrip("/").split("?")[0] or "db"


def _derive_catalogue_pattern(connection_string: str, model: str, out_dir: str) -> str:
    """
    Derive the catalogue output path pattern (no index, no extension).

    The actual filename is determined at write time so that concurrent runs
    never collide:  <out_dir>/<model>/<db_stem>_catalogue

    sqlite:///data/solar_wind.db  →  <out_dir>/gemini-2.5-flash/solar_wind_catalogue
    postgresql://.../mydb         →  <out_dir>/gemini-2.5-flash/mydb_catalogue
    """
    db_name = _db_stem(connection_string)
    model_dir = Path(out_dir) / _model_folder_name(model)
    return str(model_dir / f"{db_name}_catalogue")


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
        metavar="OUTPUT_DIR",
        help="Output directory for catalogue files (overrides SC_OUTPUT_DIR from .env)",
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
      schematica --db path/to/mydb.db --out path/to/output/dir
    """
    args = _parse_args()

    _assign_model(args)

    connection_string = _to_connection_string(args.db)

    prompt_readonly_confirmation(connection_string, skip=args.skip_ro_check)

    out_dir = args.out or os.environ.get("SC_OUTPUT_DIR")
    if not out_dir:
        print(
            "\nError: no output directory set. "
            "Add SC_OUTPUT_DIR to .env or pass --out <dir>.",
            file=sys.stderr,
        )
        sys.exit(1)
    out_path = _derive_catalogue_pattern(connection_string, agent._config.model, out_dir)
    print(f"Output → {Path(out_path).parent}/", file=sys.stderr)

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
