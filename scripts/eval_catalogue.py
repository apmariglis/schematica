"""
eval_catalogue.py — Catalogue Quality Evaluator

Runs every SQL metric in a data_catalogue.json against the live database
and produces a structured report covering:

  1. SQL validity        — does each query run without error?
  2. Shape               — does it return exactly 2 columns?
  3. Row count           — how many data points?
  4. Value quality       — null rate, numeric range
  5. Date range accuracy — does actual min/max match declared time_range?
  6. Table coverage      — are all DB tables represented in the catalogue?
  7. Confidence review   — high-confidence metrics flagged for manual spot-check

Usage:
  uv run python scripts/eval_catalogue.py \\
      --catalogue data/solar_wind_catalogue.json \\
      --db sqlite:///data/solar_wind_co.db

  # Output a JSON report instead of the default rich table:
  uv run python scripts/eval_catalogue.py \\
      --catalogue data/solar_wind_catalogue.json \\
      --db sqlite:///data/solar_wind_co.db \\
      --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from sqlalchemy import create_engine

from schematica.eval import (
    MetricResult, FactResult,
    evaluate_metric, evaluate_fact,
    check_duplicate_sql,
    WARN_NULL_RATE, WARN_MIN_ROWS,
)

console = Console()


def check_table_coverage(catalogue: dict, engine) -> list[str]:
    """Return tables in the DB that are not mentioned in any metric's tables_used."""
    from sqlalchemy import inspect as sqla_inspect
    insp = sqla_inspect(engine)
    db_tables = set(insp.get_table_names())

    mentioned = set()
    for m in catalogue.get("measurable_metrics", []):
        for t in m.get("tables_used", []):
            mentioned.add(t)
    for f in catalogue.get("queryable_facts", []):
        for t in f.get("tables_used", []):
            mentioned.add(t)
    # Also check catalogue tables section
    for t in catalogue.get("tables", []):
        mentioned.add(t["name"])

    return sorted(db_tables - mentioned)


# ── schema validation ──────────────────────────────────────────────────────────

# Fields that must be present and non-empty in a valid catalogue
_REQUIRED_CATALOGUE_FIELDS: list[tuple[str, str]] = [
    ("description",        "One-sentence dataset description (regenerate catalogue to add)"),
]


def check_catalogue_schema(catalogue: dict) -> list[tuple[str, str]]:
    """
    Return a list of (field, reason) for expected catalogue fields that are
    absent or empty. An empty list means the catalogue is structurally complete.
    """
    issues = []
    for field, reason in _REQUIRED_CATALOGUE_FIELDS:
        value = catalogue.get(field)
        if not value:
            issues.append((field, reason))
    return issues


# ── rich output ────────────────────────────────────────────────────────────────

STATUS_STYLE = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}
CONF_STYLE   = {"high": "green", "medium": "yellow", "low": "red"}


_WARN_CODE_DESCRIPTIONS = {
    "high_nulls":       "Value column has >10% NULL entries",
    "sparse":           "Fewer than 3 rows — not enough data for a metric",
    "date_mismatch":    "Actual data range falls outside the declared time_range",
    "extra_cols":       "Query returns more than 2 columns; metrics must return exactly date + value",
    "zero_rows":        "Query ran without error but returned 0 rows",
    "period_boundary":  "time_range start/end does not align to the metric's granularity boundary (e.g. monthly metric should start/end on the first day of a month)",
    "non_date_col":     "First column cannot be parsed as dates (>20% unparseable) — may be a row-number or label column",
    "constant_values":  "Value column returns the same number every period — not a useful trend metric",
    "duplicate_sql":    "SQL is identical to another metric in this catalogue (redundant entry)",
}

_STATUS_DESCRIPTIONS = {
    "PASS": "[green]✓ PASS[/green]  SQL runs · exactly 2 cols · ≥3 rows · <10% nulls · date range matches",
    "WARN": "[yellow]⚠ WARN[/yellow]  SQL runs but one or more issues found (see Notes column)",
    "FAIL": "[red]✗ FAIL[/red]  SQL error or fewer than 2 columns returned",
}

_CONF_DESCRIPTIONS = {
    "high":   "[green]high[/green]    Column names and types are unambiguous",
    "medium": "[yellow]medium[/yellow] Inferred from naming conventions or sample values",
    "low":    "[red]low[/red]     Significant uncertainty — agent made assumptions; review manually",
}


def print_legend(
    results: list[MetricResult],
    fact_results: list[FactResult],
    target_console: Console | None = None,
) -> None:
    """Print a legend explaining only the statuses and codes that appear in this run."""
    c = target_console or console

    # Collect which statuses appear
    all_results = [*results, *fact_results]
    seen_statuses = {r.status for r in all_results}

    # Collect which warn codes appear (r.error is comma-separated codes for WARN)
    seen_codes: set[str] = set()
    for r in all_results:
        if r.status == "WARN" and r.error:
            for code in r.error.split(", "):
                code = code.strip()
                if code in _WARN_CODE_DESCRIPTIONS:
                    seen_codes.add(code)

    # Check which confidence levels appear
    seen_conf = {r.confidence for r in results if hasattr(r, "confidence")}

    lines = ["[bold]Legend[/bold]\n"]

    lines.append("[bold]Status[/bold]")
    for status in ("PASS", "WARN", "FAIL"):
        if status in seen_statuses:
            lines.append(f"  {_STATUS_DESCRIPTIONS[status]}")

    if seen_codes:
        lines.append("\n[bold]Warning codes (Notes column)[/bold]")
        for code in sorted(seen_codes):
            lines.append(f"  [yellow]{code}[/yellow]  {_WARN_CODE_DESCRIPTIONS[code]}")

    if seen_conf - {"high"}:  # only explain if medium or low appear
        lines.append("\n[bold]Conf[/bold]")
        for conf in ("high", "medium", "low"):
            if conf in seen_conf:
                lines.append(f"  {_CONF_DESCRIPTIONS[conf]}")

    c.print(Panel("\n".join(lines), title="Legend", border_style="dim", padding=(0, 1)))


def print_report(results: list[MetricResult], fact_results: list[FactResult],
                 uncovered_tables: list[str], catalogue: dict,
                 schema_issues: list[tuple[str, str]] | None = None) -> None:
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_warn = sum(1 for r in results if r.status == "WARN")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    fn_pass = sum(1 for r in fact_results if r.status == "PASS")
    fn_warn = sum(1 for r in fact_results if r.status == "WARN")
    fn_fail = sum(1 for r in fact_results if r.status == "FAIL")

    summary_text = (
        f"[bold]Metrics evaluated:[/bold]       {len(results)}\n"
        f"  [green]PASS[/green] {n_pass}   [yellow]WARN[/yellow] {n_warn}   [red]FAIL[/red] {n_fail}\n"
        f"[bold]Queryable facts evaluated:[/bold] {len(fact_results)}\n"
        f"  [green]PASS[/green] {fn_pass}   [yellow]WARN[/yellow] {fn_warn}   [red]FAIL[/red] {fn_fail}\n"
        f"Catalogue time coverage: "
        f"{catalogue.get('time_coverage', {}).get('start', '?')} → "
        f"{catalogue.get('time_coverage', {}).get('end', '?')}"
    )
    if uncovered_tables:
        summary_text += f"\n[yellow]Tables with no metrics or facts:[/yellow] {', '.join(uncovered_tables)}"
    if schema_issues:
        for field, reason in schema_issues:
            summary_text += f"\n[red]Missing catalogue field '{field}':[/red] {reason}"

    console.print(Panel(summary_text, title="Catalogue Evaluation", border_style="cyan", padding=(0, 1)))
    console.print()

    tbl = Table(show_header=True, header_style="bold", expand=True)
    tbl.add_column("Status",   width=6)
    tbl.add_column("Metric",   min_width=32, no_wrap=False)
    tbl.add_column("Conf",     width=6)
    tbl.add_column("Gran",     width=10)
    tbl.add_column("Unit",     width=10)
    tbl.add_column("Rows",     width=6, justify="right")
    tbl.add_column("Null%",    width=6, justify="right")
    tbl.add_column("Value range",    width=22)
    tbl.add_column("Declared start→end", width=24)
    tbl.add_column("Actual start→end",   width=24)
    tbl.add_column("Notes",    min_width=20, no_wrap=False)

    for r in results:
        st_style  = STATUS_STYLE.get(r.status, "")
        conf_style = CONF_STYLE.get(r.confidence, "")

        null_pct = f"{r.null_rate * 100:.1f}%" if r.sql_ok and r.n_rows > 0 else "—"
        val_range = (
            f"{r.value_min:,.1f} – {r.value_max:,.1f}"
            if r.value_min is not None else "—"
        )
        declared_range = f"{r.declared_start[:7]} → {r.declared_end[:7]}" if r.declared_start else "—"
        actual_range   = (
            f"{r.actual_start[:7]} → {r.actual_end[:7]}"
            if r.actual_start else "—"
        )
        date_flag = "" if r.date_range_ok else " ⚠"

        notes = r.error or "ok"
        if r.duplicate_of:
            notes = notes.replace("duplicate_sql", f"duplicate_sql({r.duplicate_of})")

        tbl.add_row(
            Text(r.status, style=st_style),
            r.name,
            Text(r.confidence, style=conf_style),
            r.granularity,
            r.unit,
            str(r.n_rows) if r.sql_ok else "—",
            null_pct,
            val_range,
            declared_range,
            actual_range + date_flag,
            notes,
        )

    console.print(tbl)

    # Queryable facts table
    if fact_results:
        console.print()
        ftbl = Table(show_header=True, header_style="bold", expand=True, title="Queryable Facts")
        ftbl.add_column("Status", width=6)
        ftbl.add_column("Fact",   min_width=32, no_wrap=False)
        ftbl.add_column("Rows",   width=6, justify="right")
        ftbl.add_column("Cols",   width=6, justify="right")
        ftbl.add_column("Notes",  min_width=20, no_wrap=False)

        for r in fact_results:
            st_style = STATUS_STYLE.get(r.status, "")
            ftbl.add_row(
                Text(r.status, style=st_style),
                r.name,
                str(r.n_rows) if r.sql_ok else "—",
                str(r.n_cols) if r.sql_ok else "—",
                r.error or "ok",
            )
        console.print(ftbl)

    # Data quality notes
    dqn = catalogue.get("data_quality_notes", [])
    if dqn:
        console.print()
        console.print("[bold]Agent data quality notes:[/bold]")
        for note in dqn:
            console.print(f"  • {note}", highlight=False)

    console.print()
    print_legend(results, fact_results)


def print_json_report(results: list[MetricResult], fact_results: list[FactResult],
                      uncovered_tables: list[str], catalogue: dict) -> None:
    report = {
        "summary": {
            "metrics_total":  len(results),
            "metrics_pass":   sum(1 for r in results if r.status == "PASS"),
            "metrics_warn":   sum(1 for r in results if r.status == "WARN"),
            "metrics_fail":   sum(1 for r in results if r.status == "FAIL"),
            "facts_total":    len(fact_results),
            "facts_pass":     sum(1 for r in fact_results if r.status == "PASS"),
            "facts_warn":     sum(1 for r in fact_results if r.status == "WARN"),
            "facts_fail":     sum(1 for r in fact_results if r.status == "FAIL"),
            "uncovered_tables": uncovered_tables,
        },
        "metrics": [
            {
                "name":           r.name,
                "status":         r.status,
                "confidence":     r.confidence,
                "granularity":    r.granularity,
                "unit":           r.unit,
                "n_rows":         r.n_rows,
                "null_rate":      round(r.null_rate, 4),
                "value_min":      r.value_min,
                "value_max":      r.value_max,
                "declared_start": r.declared_start,
                "declared_end":   r.declared_end,
                "actual_start":   r.actual_start,
                "actual_end":     r.actual_end,
                "date_range_ok":  r.date_range_ok,
                "date_col_ok":    r.date_col_ok,
                "duplicate_of":   r.duplicate_of,
                "error":          r.error,
            }
            for r in results
        ],
        "queryable_facts": [
            {
                "name":   r.name,
                "status": r.status,
                "n_rows": r.n_rows,
                "n_cols": r.n_cols,
                "error":  r.error,
            }
            for r in fact_results
        ],
    }
    print(json.dumps(report, indent=2))


# ── entry point ────────────────────────────────────────────────────────────────

_KNOWN_SCHEMES = ("sqlite:///", "postgresql://", "mysql://", "mssql://", "oracle://")


def _to_connection_string(db: str) -> str:
    if any(db.startswith(s) for s in _KNOWN_SCHEMES):
        return db
    return f"sqlite:///{db}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a data_catalogue.json against a live database."
    )
    parser.add_argument("--db", required=True, metavar="DB",
                        help="Database file path (e.g. ./data/mydb.db) or SQLAlchemy connection string")
    parser.add_argument("--catalogue", required=True, metavar="PATH",
                        help="Path to *_catalogue.json")
    parser.add_argument("--json", action="store_true",
                        help="Output a machine-readable JSON report instead of the rich table")
    args = parser.parse_args()

    catalogue_path = Path(args.catalogue)
    if not catalogue_path.exists():
        console.print(f"[red]Catalogue not found: {catalogue_path}[/red]")
        sys.exit(1)

    db_string = _to_connection_string(args.db)

    with open(catalogue_path) as f:
        catalogue = json.load(f)

    metrics = catalogue.get("measurable_metrics", [])
    facts   = catalogue.get("queryable_facts", [])

    if not metrics and not facts:
        console.print("[red]No measurable_metrics or queryable_facts found in catalogue.[/red]")
        sys.exit(1)

    engine = create_engine(db_string)
    icon = {"PASS": "[green]✓[/green]", "WARN": "[yellow]⚠[/yellow]", "FAIL": "[red]✗[/red]"}

    if not args.json:
        console.print(f"[dim]Evaluating {len(metrics)} metrics and {len(facts)} facts from {catalogue_path.name}…[/dim]\n")

    # Pre-compute duplicate SQL mapping so we can stamp results after evaluation
    duplicates = check_duplicate_sql(metrics)

    results = []
    for metric in metrics:
        r = evaluate_metric(engine, metric)
        if metric["name"] in duplicates:
            r.duplicate_of = duplicates[metric["name"]]
            codes = [c for c in r.error.split(", ") if c] if r.error else []
            if "duplicate_sql" not in codes:
                codes.append("duplicate_sql")
            r.error  = ", ".join(codes)
            r.status = "WARN" if r.status == "PASS" else r.status
        results.append(r)
        if not args.json:
            console.print(f"  {icon[r.status]} {r.name}", highlight=False)

    fact_results = []
    for fact in facts:
        r = evaluate_fact(engine, fact)
        fact_results.append(r)
        if not args.json:
            console.print(f"  {icon[r.status]} [dim](fact)[/dim] {r.name}", highlight=False)

    uncovered = check_table_coverage(catalogue, engine)
    schema_issues = check_catalogue_schema(catalogue)

    if args.json:
        print_json_report(results, fact_results, uncovered, catalogue)
    else:
        console.print()
        print_report(results, fact_results, uncovered, catalogue, schema_issues)


if __name__ == "__main__":
    main()
