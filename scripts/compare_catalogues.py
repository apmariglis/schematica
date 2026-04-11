"""
compare_catalogues.py — Cross-model catalogue quality comparison.

Accepts explicit database connections and catalogue files (or a directory),
matches each catalogue to its database, and produces a side-by-side quality
report.

Usage:
  # scan a whole data directory for all catalogues
  uv run python scripts/compare_catalogues.py \\
      --dbs path/to/mydb.db \\
      --catalogues data/

  # multiple databases, one data directory
  uv run python scripts/compare_catalogues.py \\
      --dbs path/to/db1.db postgresql://user:pw@host/db2 \\
      --catalogues data/

  # catalogues scattered across different folders
  uv run python scripts/compare_catalogues.py \\
      --dbs path/to/mydb.db \\
      --catalogues runs/2025-01-10/mydb_catalogue_1.json \\
                   runs/2025-02-14/mydb_catalogue_1.json \\
                   team/alice/mydb_catalogue_1.json

  # multiple databases, catalogues scattered across different folders
  uv run python scripts/compare_catalogues.py \\
      --dbs path/to/db1.db path/to/db2.db \\
      --catalogues runs/gemini/db1_catalogue_1.json \\
                   runs/claude/db1_catalogue_1.json \\
                   archive/db2_catalogue_1.json \\
                   team/alice/db2_catalogue_1.json

Each catalogue JSON is matched to its database by comparing the stem of the
`connection` field stored inside the catalogue against the stem of each --dbs
entry. The model label is taken from the catalogue file's parent directory name.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from schematica.pricing import format_cost
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from sqlalchemy import inspect as sqla_inspect
from schematica.db import make_readonly_engine, prompt_readonly_confirmation

from schematica.eval import (
    evaluate_metric,
    evaluate_fact,
    check_duplicate_sql,
)

console = Console()


# ── Discovery ──────────────────────────────────────────────────────────────────

_CATALOGUE_INDEX_RE = re.compile(r"_catalogue(?:_(\d+))?\.json$")


def _collect_catalogue_paths(sources: list[str]) -> list[Path]:
    """
    Expand a mixed list of directories and explicit JSON file paths into a flat
    list of catalogue JSON paths. Directories are scanned one level deep for
    *_catalogue*.json files in their immediate subfolders.
    """
    paths: list[Path] = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            for subdir in sorted(p.iterdir()):
                if subdir.is_dir():
                    paths.extend(sorted(subdir.glob("*_catalogue*.json")))
        elif p.suffix == ".json":
            paths.append(p)
        else:
            console.print(f"[yellow]Skipping unrecognised source: {src}[/yellow]")
    return paths


def _catalogue_entry(path: Path) -> dict | None:
    """
    Read a catalogue JSON and return an entry dict:
      {"model": str, "index": int, "path": Path, "db_stem": str}

    Returns None if the file cannot be read or lacks a connection field.
    The model label is the parent directory name.
    The db_stem is derived from the `connection` field stored in the catalogue.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        console.print(f"[yellow]Could not read {path}: {exc}[/yellow]")
        return None

    connection = data.get("connection", "")
    if not connection:
        console.print(f"[yellow]No connection field in {path} — skipping[/yellow]")
        return None

    m = _CATALOGUE_INDEX_RE.search(path.name)
    index = int(m.group(1)) if m and m.group(1) else 1

    return {
        "model":   path.parent.name,
        "index":   index,
        "path":    path,
        "db_stem": _db_stem(connection),
    }


def group_catalogues_by_db(sources: list[str]) -> dict[str, list[dict]]:
    """
    Return {db_stem: [entry, ...]} built from all catalogue files found in sources.
    sources may be directories or explicit JSON file paths.
    """
    result: dict[str, list[dict]] = defaultdict(list)
    for path in _collect_catalogue_paths(sources):
        entry = _catalogue_entry(path)
        if entry:
            result[entry["db_stem"]].append(entry)
    return dict(result)


_KNOWN_SCHEMES = ("sqlite:///", "postgresql://", "mysql://", "mssql://", "oracle://")


def _to_connection_string(db: str) -> str:
    if any(db.startswith(s) for s in _KNOWN_SCHEMES):
        return db
    return f"sqlite:///{db}"


def _db_stem(db: str) -> str:
    """Extract the database name stem from a file path or connection string."""
    if "://" in db:
        path = db.split("://", 1)[1].lstrip("/").split("?")[0]
        return Path(path).stem
    return Path(db).stem


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _count_db_tables(engine) -> int:
    return len(sqla_inspect(engine).get_table_names())


def _count_covered_tables(catalogue: dict, engine) -> int:
    db_tables = set(sqla_inspect(engine).get_table_names())
    mentioned: set[str] = set()
    for m in catalogue.get("measurable_metrics", []):
        mentioned.update(m.get("tables_used", []))
    for f in catalogue.get("queryable_facts", []):
        mentioned.update(f.get("tables_used", []))
    for t in catalogue.get("tables", []):
        mentioned.add(t["name"])
    return len(db_tables & mentioned)


def _count_join_metrics(catalogue: dict) -> int:
    """Metrics whose SQL touches more than one table."""
    return sum(
        1 for m in catalogue.get("measurable_metrics", [])
        if len(m.get("tables_used", [])) > 1
    )


# ── LLM judge ──────────────────────────────────────────────────────────────────

_JUDGE_MODEL      = "gemini/gemini-2.5-flash"
_JUDGE_BATCH_SIZE = 10   # metrics per API call

_JUDGE_SYSTEM = """\
You are a strict SQL correctness judge. You will be given a database schema and \
a list of metrics, each with a name, description, and SQL query. Your task is to \
rate how accurately the SQL implements what the name and description say.

Scoring rubric (1–5):
5 — SQL precisely implements the name/description: correct columns, aggregation, \
    grouping, and filters
4 — SQL correctly implements the intent with only trivial caveats (e.g. minor \
    column alias difference)
3 — SQL roughly captures the intent but has a notable issue (wrong aggregation, \
    missing filter, slightly off column)
2 — SQL partially addresses the intent but has significant problems that would \
    produce misleading results
1 — SQL does not implement what the name/description says, or is factually wrong

Respond with a JSON array ONLY — no prose, no markdown fences. Each element must \
have exactly these keys: "name" (string), "score" (integer 1-5), "reason" (one \
concise sentence explaining the score).
"""


def _db_schema_text(engine) -> str:
    """Compact schema: one line per table listing column names."""
    insp = sqla_inspect(engine)
    lines = []
    for table in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns(table)]
        lines.append(f"{table}({', '.join(cols)})")
    return "\n".join(lines)


def _parse_judge_response(text: str) -> list[dict]:
    """Extract a JSON array from the response, tolerating markdown fences and truncation."""
    text = text.strip()
    # Strip ```json ... ``` fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Try clean parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed = next(v for v in parsed.values() if isinstance(v, list))
        return parsed
    except json.JSONDecodeError:
        pass

    # Response may be truncated — recover complete objects from the array
    items = []
    for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"score"\s*:\s*(\d)[^{}]*\}', text):
        try:
            items.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            items.append({"name": m.group(1), "score": int(m.group(2)), "reason": ""})
    return items


def run_judge(metrics: list[dict], schema_text: str,
              usage: dict) -> dict[str, dict]:
    """
    Score each metric for semantic correctness. Returns {metric_name: {"score": N, "reason": str}}.
    Model names are never included in the prompt — the judge evaluates SQL on merit alone.
    Accumulates token counts into the provided usage dict (keys: input_tokens, output_tokens).
    """
    try:
        import litellm
    except ImportError:
        raise SystemExit("litellm is required for --judge. Install with: uv sync --extra litellm")

    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is required for --judge.")

    results: dict[str, dict] = {}

    for i in range(0, len(metrics), _JUDGE_BATCH_SIZE):
        batch = metrics[i : i + _JUDGE_BATCH_SIZE]
        items = "\n\n".join(
            f"name: {m['name']}\n"
            f"description: {m.get('description', '')}\n"
            f"sql: {m.get('sql', '')}"
            for m in batch
        )
        user_msg = f"Database schema:\n{schema_text}\n\nMetrics to evaluate:\n{items}"

        try:
            resp = litellm.completion(
                model=_JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=8192,
                temperature=0,
            )
            usage["input_tokens"]  += resp.usage.prompt_tokens
            usage["output_tokens"] += resp.usage.completion_tokens
            parsed = _parse_judge_response(resp.choices[0].message.content)
            for item in parsed:
                results[item["name"]] = {
                    "score":  float(item.get("score", 3)),
                    "reason": item.get("reason", ""),
                }
        except Exception as exc:
            console.print(f"[yellow]Judge batch {i//10 + 1} failed: {exc}[/yellow]")

    return results


def _avg_judge_score(judge_scores: dict[str, dict]) -> float | None:
    scores = [v["score"] for v in judge_scores.values() if "score" in v]
    return round(sum(scores) / len(scores), 2) if scores else None


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_catalogue_entry(entry: dict, engine,
                             judge_scores: dict[str, dict] | None = None) -> dict:
    """
    Run eval on a single catalogue entry and return a flat summary dict.
    judge_scores, if provided, is {metric_name: {"score": N, "reason": str}}.
    """
    with open(entry["path"]) as f:
        catalogue = json.load(f)

    metrics = catalogue.get("measurable_metrics", [])
    facts   = catalogue.get("queryable_facts", [])

    duplicates = check_duplicate_sql(metrics)

    metric_results = []
    for metric in metrics:
        r = evaluate_metric(engine, metric)
        if metric["name"] in duplicates:
            r.duplicate_of = duplicates[metric["name"]]
            codes = [c for c in r.error.split(", ") if c] if r.error else []
            if "duplicate_sql" not in codes:
                codes.append("duplicate_sql")
            r.error  = ", ".join(codes)
            r.status = "WARN" if r.status == "PASS" else r.status
        metric_results.append(r)

    fact_results = [evaluate_fact(engine, f) for f in facts]

    n_total   = len(metric_results)
    n_pass    = sum(1 for r in metric_results if r.status == "PASS")
    n_warn    = sum(1 for r in metric_results if r.status == "WARN")
    n_fail    = sum(1 for r in metric_results if r.status == "FAIL")
    n_dup     = sum(1 for r in metric_results if r.duplicate_of)
    n_const   = sum(1 for r in metric_results if "constant_values" in (r.error or ""))
    n_joins   = _count_join_metrics(catalogue)
    n_covered = _count_covered_tables(catalogue, engine)
    n_db_tabs = _count_db_tables(engine)
    n_facts   = len(fact_results)
    n_facts_pass = sum(1 for r in fact_results if r.status == "PASS")

    # Confidence: % of metrics declared as high-confidence
    n_high_conf = sum(1 for r in metric_results if r.confidence == "high")
    conf_rate   = n_high_conf / n_total if n_total else 0.0

    # Date range accuracy: % of metrics whose actual range matches declared
    date_ok_rate = (
        sum(1 for r in metric_results if r.sql_ok and r.date_range_ok) / n_total
        if n_total else 0.0
    )

    # Median row count across successfully-executed metrics
    row_counts = [r.n_rows for r in metric_results if r.sql_ok]
    median_rows = float(sorted(row_counts)[len(row_counts) // 2]) if row_counts else 0.0

    # Average declared time span in months
    def _span_months(m: dict) -> float | None:
        tr = m.get("time_range", {})
        s, e = tr.get("start", ""), tr.get("end", "")
        if not s or not e:
            return None
        try:
            from datetime import date
            sy, sm = int(s[:4]), int(s[5:7])
            ey, em = int(e[:4]), int(e[5:7])
            return (ey - sy) * 12 + (em - sm)
        except (ValueError, IndexError):
            return None

    spans = [v for m in metrics if (v := _span_months(m)) is not None]
    avg_span = sum(spans) / len(spans) if spans else 0.0

    pass_rate     = n_pass / n_total if n_total else 0.0
    coverage_rate = n_covered / n_db_tabs if n_db_tabs else 0.0
    noise_ratio   = (n_dup + n_const) / n_total if n_total else 0.0
    # Composite: correctness × breadth × (1 − noise), expressed as 0–100
    composite = pass_rate * coverage_rate * (1.0 - noise_ratio) * 100.0

    return {
        "model":            entry["model"],
        "index":            entry["index"],
        "path":             str(entry["path"]),
        "n_metrics":        n_total,
        "n_pass":           n_pass,
        "n_warn":           n_warn,
        "n_fail":           n_fail,
        "n_duplicates":     n_dup,
        "n_constants":      n_const,
        "n_joins":          n_joins,
        "n_facts":          n_facts,
        "n_facts_pass":     n_facts_pass,
        "n_covered_tables": n_covered,
        "n_db_tables":      n_db_tabs,
        "pass_rate":        round(pass_rate, 4),
        "coverage_rate":    round(coverage_rate, 4),
        "noise_ratio":      round(noise_ratio, 4),
        "conf_rate":        round(conf_rate, 4),
        "date_ok_rate":     round(date_ok_rate, 4),
        "median_rows":      round(median_rows, 1),
        "avg_span_months":  round(avg_span, 1),
        "composite":        round(composite, 1),
        "semantic_score":   _avg_judge_score(judge_scores) if judge_scores else None,
        "judge_details":    judge_scores or {},
        "combined":         round(
            composite * (_avg_judge_score(judge_scores) / 5.0), 1
        ) if judge_scores and _avg_judge_score(judge_scores) is not None else None,
    }


# ── Rich output ────────────────────────────────────────────────────────────────

_SCORE_STYLE = {
    "high":   "green",
    "medium": "yellow",
    "low":    "red",
}


def _score_style(composite: float) -> str:
    if composite >= 70:
        return "green"
    if composite >= 40:
        return "yellow"
    return "red"


def _semantic_style(score: float) -> str:
    if score >= 4.0:
        return "green"
    if score >= 3.0:
        return "yellow"
    return "red"


def print_comparison(db_name: str, rows: list[dict]) -> None:
    has_judge = any(r.get("semantic_score") is not None for r in rows)
    rows_sorted = sorted(rows, key=lambda r: -(r.get("combined") or 0)
                         if has_judge else -r["composite"])

    tbl = Table(
        show_header=True, header_style="bold", expand=True,
        title=f"[bold cyan]{db_name}[/bold cyan]  ({rows[0]['n_db_tables']} tables in DB)",
    )
    tbl.add_column("Model",     min_width=20, no_wrap=True)
    tbl.add_column("#",         width=3,  justify="right")
    tbl.add_column("Metrics",   width=7,  justify="right")
    tbl.add_column("Pass%",     width=6,  justify="right")
    tbl.add_column("WARN",      width=5,  justify="right")
    tbl.add_column("FAIL",      width=5,  justify="right")
    tbl.add_column("Dup",       width=4,  justify="right")
    tbl.add_column("Const",     width=5,  justify="right")
    tbl.add_column("Joins%",    width=6,  justify="right")
    tbl.add_column("Cover%",    width=6,  justify="right")
    tbl.add_column("Conf%",     width=6,  justify="right")
    tbl.add_column("DateOK%",   width=7,  justify="right")
    tbl.add_column("MedRows",   width=7,  justify="right")
    tbl.add_column("AvgSpan",   width=7,  justify="right")
    tbl.add_column("Facts",     width=5,  justify="right")
    tbl.add_column("Score",     width=6,  justify="right")
    if has_judge:
        tbl.add_column("Semantic",  width=8,  justify="right")
        tbl.add_column("Combined",  width=8,  justify="right")

    for r in rows_sorted:
        pass_pct    = f"{r['pass_rate']*100:.0f}%"
        join_pct    = f"{r['n_joins']/r['n_metrics']*100:.0f}%" if r["n_metrics"] else "—"
        cover_pct   = f"{r['coverage_rate']*100:.0f}%"
        conf_pct    = f"{r['conf_rate']*100:.0f}%"
        dateok_pct  = f"{r['date_ok_rate']*100:.0f}%"
        med_rows    = str(int(r["median_rows"])) if r["median_rows"] else "—"
        avg_span    = f"{r['avg_span_months']:.0f}mo" if r["avg_span_months"] else "—"

        row_cells = [
            r["model"],
            str(r["index"]),
            str(r["n_metrics"]),
            pass_pct,
            str(r["n_warn"]),
            str(r["n_fail"]),
            str(r["n_duplicates"]) if r["n_duplicates"] else "—",
            str(r["n_constants"])  if r["n_constants"]  else "—",
            join_pct,
            cover_pct,
            conf_pct,
            dateok_pct,
            med_rows,
            avg_span,
            f"{r['n_facts_pass']}/{r['n_facts']}",
            Text(f"{r['composite']:.1f}", style=_score_style(r["composite"])),
        ]
        if has_judge:
            sem  = r.get("semantic_score")
            comb = r.get("combined")
            row_cells.append(
                Text(f"{sem:.2f}/5", style=_semantic_style(sem)) if sem is not None
                else Text("—", style="dim")
            )
            row_cells.append(
                Text(f"{comb:.1f}", style=_score_style(comb)) if comb is not None
                else Text("—", style="dim")
            )
        tbl.add_row(*row_cells)

    console.print(tbl)
    footnote = (
        "[dim]Score = pass_rate × coverage_rate × (1 − noise_ratio) × 100  "
        "| noise = duplicates + constant-value metrics"
    )
    if has_judge:
        footnote += (
            f"  |  Semantic = avg SQL-correctness (1–5) by {_JUDGE_MODEL}"
            "  |  Combined = Score × Semantic/5"
        )
    console.print(footnote + "[/dim]\n")


def print_overall_summary(all_rows: list[dict]) -> None:
    """Print one aggregated row per model across all evaluated DBs."""
    has_judge = any(r.get("combined") is not None for r in all_rows)

    # Group by model name
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_model[r["model"]].append(r)

    def _mean(vals: list[float | None]) -> float | None:
        clean = [v for v in vals if v is not None]
        return round(sum(clean) / len(clean), 1) if clean else None

    summary_rows = []
    for model, rows in by_model.items():
        summary_rows.append({
            "model":    model,
            "n_dbs":    len(rows),
            "score":    _mean([r["composite"]     for r in rows]),
            "semantic": _mean([r.get("semantic_score") for r in rows]) if has_judge else None,
            "combined": _mean([r.get("combined")   for r in rows]) if has_judge else None,
        })

    sort_key = "combined" if has_judge else "score"
    summary_rows.sort(key=lambda r: -(r[sort_key] or 0))

    tbl = Table(show_header=True, header_style="bold", expand=False,
                title="[bold]Overall model ranking[/bold]")
    tbl.add_column("Model",    min_width=20, no_wrap=True)
    tbl.add_column("DBs",      width=4,  justify="right")
    tbl.add_column("Score",    width=7,  justify="right")
    if has_judge:
        tbl.add_column("Semantic", width=8,  justify="right")
        tbl.add_column("Combined", width=8,  justify="right")

    for r in summary_rows:
        row_cells = [
            r["model"],
            str(r["n_dbs"]),
            Text(f"{r['score']:.1f}", style=_score_style(r["score"])) if r["score"] is not None
            else Text("—", style="dim"),
        ]
        if has_judge:
            sem  = r["semantic"]
            comb = r["combined"]
            row_cells.append(
                Text(f"{sem:.2f}/5", style=_semantic_style(sem)) if sem is not None
                else Text("—", style="dim")
            )
            row_cells.append(
                Text(f"{comb:.1f}", style=_score_style(comb)) if comb is not None
                else Text("—", style="dim")
            )
        tbl.add_row(*row_cells)

    console.print(tbl)


def print_json_output(all_results: dict[str, list[dict]]) -> None:
    print(json.dumps(all_results, indent=2))


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare catalogues generated by different models for the same database."
    )
    parser.add_argument(
        "--dbs", nargs="+", required=True, metavar="PATH",
        help=(
            "One or more database file paths or SQLAlchemy connection strings. "
            "Each catalogue is matched to its database via the connection field "
            "stored inside the catalogue JSON."
        ),
    )
    parser.add_argument(
        "--catalogues", nargs="+", required=True, metavar="PATH",
        help=(
            "One or more catalogue JSON files, or a directory whose immediate "
            "subfolders are scanned for *_catalogue*.json files."
        ),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit a machine-readable JSON report instead of the rich table",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help=(
            f"Run an LLM semantic judge ({_JUDGE_MODEL}) that scores each metric's "
            "SQL against its name/description (1–5). Requires GOOGLE_API_KEY."
        ),
    )
    parser.add_argument("--skip-ro-check", action="store_true",
                        help="Skip the read-only user confirmation prompt (for CI / automated use)")
    args = parser.parse_args()

    started_at = time.monotonic()

    # Build stem → connection string map for every supplied database
    db_by_stem: dict[str, str] = {}
    for db in args.dbs:
        conn_str = _to_connection_string(db)
        stem = _db_stem(db)
        db_by_stem[stem] = conn_str
        prompt_readonly_confirmation(conn_str, skip=args.skip_ro_check)

    # Group catalogues by the db stem stored in their connection field
    all_catalogues = group_catalogues_by_db(args.catalogues)

    # Keep only stems that have a matching database in --dbs
    matched: dict[str, list[dict]] = {}
    for stem, entries in all_catalogues.items():
        if stem in db_by_stem:
            matched[stem] = entries
        else:
            console.print(
                f"[yellow]No matching --dbs entry for catalogue db '{stem}' — skipping[/yellow]"
            )

    if not matched:
        console.print("[red]No catalogues could be matched to the supplied databases.[/red]")
        sys.exit(1)

    json_output: dict[str, list[dict]] = {}
    judge_usage  = {"input_tokens": 0, "output_tokens": 0}
    all_rows: list[dict] = []

    for db_stem, entries in sorted(matched.items()):
        if not args.json:
            console.print(
                f"[dim]Evaluating {len(entries)} catalogue(s) for [bold]{db_stem}[/bold]…[/dim]"
            )

        engine = make_readonly_engine(db_by_stem[db_stem])
        schema_text = _db_schema_text(engine) if args.judge else ""

        rows = []
        for entry in entries:
            if not args.json:
                console.print(
                    f"  [dim]{entry['model']} #{entry['index']}  ({entry['path']})[/dim]"
                )

            judge_scores = None
            if args.judge:
                with open(entry["path"]) as f:
                    catalogue_metrics = json.load(f).get("measurable_metrics", [])
                if not args.json:
                    console.print(
                        f"  [dim]  judging {len(catalogue_metrics)} metrics with {_JUDGE_MODEL}…[/dim]"
                    )
                judge_scores = run_judge(catalogue_metrics, schema_text, judge_usage)

            rows.append(evaluate_catalogue_entry(entry, engine, judge_scores))

        all_rows.extend(rows)

        if args.json:
            json_output[db_stem] = rows
        else:
            console.print()
            print_comparison(db_stem, rows)

    if args.json:
        print_json_output(json_output)
    else:
        if len(all_rows) > 0:
            print_overall_summary(all_rows)
    if not args.json and args.judge and (judge_usage["input_tokens"] or judge_usage["output_tokens"]):
        inp, out = judge_usage["input_tokens"], judge_usage["output_tokens"]
        elapsed = time.monotonic() - started_at
        mins, secs = divmod(int(elapsed), 60)
        elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        console.print(
            f"[dim]Judge ({_JUDGE_MODEL}): {inp:,} in + {out:,} out"
            f"  |  Cost: {format_cost(_JUDGE_MODEL, inp, out)}"
            f"  |  Elapsed: {elapsed_str}[/dim]"
        )


if __name__ == "__main__":
    main()
