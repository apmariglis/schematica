"""
output.py — Console output, iteration stats, and file writing.

All Rich rendering, print helpers, and catalogue file I/O live here.
The `console` singleton is the single output channel for the whole application.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from schematica.pricing import CACHE_READ_MULTIPLIER
from schematica.pricing import CACHE_WRITE_MULTIPLIER
from schematica.pricing import format_cost
from schematica.pricing import get_model_pricing

console = Console()

# Descriptions for warn codes that appear in Phase 3 eval results.
# Only codes that actually appear in a run are shown in the legend.
_PHASE3_WARN_LEGEND: dict[str, str] = {
    "zero_rows":        "SQL ran without error but returned 0 rows — filter condition may be wrong or data is absent",
    "sparse":           "Fewer than 3 rows returned — not enough data points for a reliable metric",
    "high_nulls":       "Value column has >10% NULL entries — may silently skew aggregations",
    "date_mismatch":    "Actual data range falls outside the declared time_range — auto-patched below",
    "extra_cols":       "Query returns more than 2 columns — metrics must return exactly date + value",
    "period_boundary":  "time_range start/end does not align to the granularity boundary (e.g. monthly → first of month) — auto-patched below",
    "constant_values":  "All periods return the same value — may indicate sparse/small source data or a SQL logic error (missing GROUP BY)",
}


def _calc_rpm(n_requests: int, elapsed_secs: float) -> float:
    """Return requests-per-minute rate; 0.0 when there is no data yet."""
    if n_requests <= 0 or elapsed_secs <= 0:
        return 0.0
    return (n_requests / elapsed_secs) * 60.0


class _RequestTracker:
    """Tracks total API requests and a per-minute window counter.

    `in_minute` resets to zero each time 60 seconds have elapsed since
    the start of the current minute window.
    """

    def __init__(self, started_at: float) -> None:
        self._started_at = started_at
        self._minute_start = started_at
        self.total = 0
        self.in_minute = 0

    def record(self, now: float) -> None:
        """Record one completed API call at wall-clock time `now`."""
        if now - self._minute_start >= 60.0:
            self._minute_start = now
            self.in_minute = 0
        self.total += 1
        self.in_minute += 1

    def rpm(self, now: float) -> float:
        """Overall requests-per-minute since the run started."""
        return _calc_rpm(self.total, now - self._started_at)


def _format_iter_stats(
    in_tokens: int,
    out_tokens: int,
    model: str,
    pricing: dict | None = None,
    # per-iteration
    iter_duration: float = 0.0,
    iter_num: int = 0,
    max_iter: int = 0,
    context_window: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    thinking_tokens: int = 0,
    # averages over all completed iterations (shown when all_iters > 0)
    all_iters: int = 0,
    # session totals (shown when session_tracker is provided)
    session_tracker: "_RequestTracker | None" = None,
    now: float = 0.0,
    session_total_in: int = 0,
    session_total_out: int = 0,
    session_total_thinking: int = 0,
    session_total_cost: float = 0.0,
    session_total_cache_create: int = 0,
    session_total_cache_read: int = 0,
) -> str:
    """Return a box-formatted stats block printed between iterations.

    Sections (all optional except current iter):
      current iter    — per-call Tokens, cost, duration, context%
      averages        — averages over all completed iterations (all_iters > 0)
      session         — cross-phase totals + llm calls/min (session_tracker provided)

    Top border embeds "1 iter = 1 LLM call". Bottom border embeds llm calls/min when session is shown.
    """
    p = get_model_pricing(model, pricing)
    iter_cost = (in_tokens * p["input"] + out_tokens * p["output"]) / 1_000_000
    if cache_creation_tokens:
        iter_cost += cache_creation_tokens * p.get("cache_write", p["input"] * CACHE_WRITE_MULTIPLIER) / 1_000_000
    if cache_read_tokens:
        iter_cost += cache_read_tokens * p.get("cache_read", p["input"] * CACHE_READ_MULTIPLIER) / 1_000_000

    def _fmt_dur(secs: float) -> str:
        if secs < 60:
            return f"{secs:.1f}s"
        m, s = divmod(int(secs), 60)
        return f"{m}m{s:02d}s"

    _SEP = " · "

    # ── current iter content ──────────────────────────────────────────────────
    effective_in = in_tokens + cache_creation_tokens + cache_read_tokens
    _fill = f"{_SEP}{effective_in / context_window * 100:.1f}% context" if context_window > 0 else ""

    def _out_str(out: int, think: int) -> str:
        if think:
            return f"{out:,} out ({out - think:,} out/{think:,} think)"
        return f"{out:,} out"

    _cache_parts = ""
    if cache_read_tokens:
        _cache_parts += f" | {cache_read_tokens:,} cached"
    if cache_creation_tokens:
        _cache_parts += f" | {cache_creation_tokens:,} cache↑"

    iter_content = (
        f"  Tokens: {in_tokens:,} in | {_out_str(out_tokens, thinking_tokens)}{_cache_parts}"
        f"{_SEP}${iter_cost:.4f}{_SEP}{_fmt_dur(iter_duration)}{_fill}  "
    )

    # ── averages over all completed iterations ────────────────────────────────
    show_averages = all_iters > 0
    if show_averages:
        avg_in       = session_total_in       // all_iters
        avg_out      = session_total_out      // all_iters
        avg_thinking = session_total_thinking // all_iters
        avg_cost     = session_total_cost     / all_iters
        _total_elapsed = (now - session_tracker._started_at) if session_tracker is not None else 0.0
        avg_dur      = _total_elapsed / all_iters
        avg_effective_in = avg_in + (session_total_cache_create + session_total_cache_read) // all_iters
        _avg_fill = f"{_SEP}{avg_effective_in / context_window * 100:.1f}% context" if context_window > 0 else ""
        avg_content = (
            f"  Tokens: {avg_in:,} in | {_out_str(avg_out, avg_thinking)}"
            f"{_SEP}${avg_cost:.4f}{_SEP}{_fmt_dur(avg_dur)}{_avg_fill}  "
        )

    # ── session totals content ────────────────────────────────────────────────
    show_session = session_tracker is not None and session_tracker.total > 0
    if show_session:
        session_elapsed = now - session_tracker._started_at
        llm_calls_min   = session_tracker.rpm(now)
        session_content = (
            f"  Tokens: {session_total_in:,} in | {_out_str(session_total_out, session_total_thinking)}"
            f"{_SEP}${session_total_cost:.4f}{_SEP}{_fmt_dur(session_elapsed)}  "
        )

    # ── section headers ───────────────────────────────────────────────────────
    _iter_label  = f"current iter {iter_num}/{max_iter}" if iter_num and max_iter else "current iter"
    _LEFT_TOP    = f"─ {_iter_label} "
    _RIGHT_TOP   = " 1 iter = 1 LLM call ─"
    _AVG_HDR     = "─ averages over all completed iterations "
    _SESSION_HDR = "─ session (accumulated) "
    _BOTTOM_LABEL = f" approx. {llm_calls_min:.1f} llm calls/min " if show_session else ""

    # ── compute inner width ───────────────────────────────────────────────────
    min_top_w = len(_LEFT_TOP) + len(_RIGHT_TOP)
    candidates = [len(iter_content), min_top_w, len(_LEFT_TOP) + 2]
    if show_averages:
        candidates += [len(avg_content), len(_AVG_HDR) + 2]
    if show_session:
        candidates += [len(session_content), len(_SESSION_HDR) + 2, len(_BOTTOM_LABEL) + 4]
    inner_w = max(candidates)

    # ── build borders ─────────────────────────────────────────────────────────
    top_fill = inner_w - len(_LEFT_TOP) - len(_RIGHT_TOP)
    top    = "╭" + _LEFT_TOP + "─" * top_fill + _RIGHT_TOP + "╮"
    if _BOTTOM_LABEL:
        _bot_left  = (inner_w - len(_BOTTOM_LABEL)) // 2
        _bot_right = inner_w - len(_BOTTOM_LABEL) - _bot_left
        bottom = "╰" + "─" * _bot_left + _BOTTOM_LABEL + "─" * _bot_right + "╯"
    else:
        bottom = "╰" + "─" * inner_w + "╯"

    def _mid(hdr: str) -> str:
        return "├" + hdr + "─" * (inner_w - len(hdr)) + "┤"

    def _row(content: str) -> str:
        return f"│{content.ljust(inner_w)}│"

    # ── assemble ──────────────────────────────────────────────────────────────
    lines = [top, _row(iter_content)]
    if show_averages:
        lines += [_mid(_AVG_HDR), _row(avg_content)]
    if show_session:
        lines += [_mid(_SESSION_HDR), _row(session_content)]
    lines.append(bottom)
    return "\n".join(lines)



def _print_header(connection_string: str, out_path: str, model: str, cache: bool) -> None:
    model_line = model
    if model.startswith("anthropic/"):
        model_line += "  [dim](cache: on)[/dim]" if cache else "  [dim](cache: off — set SC_CACHE=true to enable)[/dim]"
    console.print(Panel(
        f"[bold]Schematica[/bold]\n"
        f"[dim]Source:[/dim]  {connection_string}\n"
        f"[dim]Output:[/dim]  {out_path}\n"
        f"[dim]Model:[/dim]   {model_line}",
        border_style="cyan",
        padding=(0, 1),
    ))


def _print_schema_summary(snapshot: dict) -> None:
    t = Text()
    t.append("Schema snapshot (pre-computed against full DB — row counts, min/max, and null rates are exact):\n", style="bold")
    for tbl in snapshot["tables"]:
        t.append(f"  {tbl['name']}", style="cyan")
        t.append(f"  ({tbl['row_count']:,} rows, {len(tbl['columns'])} cols)\n", style="dim")
    console.print(t)


def _print_schema_detail(schema_text: str) -> None:
    console.print(Panel(
        f"[dim]{schema_text}[/dim]",
        title="Schema sent to LLM",
        border_style="dim",
        padding=(0, 1),
    ))


def _print_query(
    sql: str,
    reason: str,
    tables: list[str],
    columns: list[str],
    plain_language: str,
    result: str,
    table_columns: dict[str, list[str]],
) -> None:
    used = set(columns)

    # For each table: used columns first (bold), then remaining columns
    col_parts: list[str] = []
    for tbl in tables:
        all_cols = table_columns.get(tbl, [])
        ordered = [c for c in all_cols if c in used] + [c for c in all_cols if c not in used]
        for c in ordered:
            if c in used:
                col_parts.append(f"[bold]{c}[/bold]")
            else:
                col_parts.append(f"[dim]{c}[/dim]")

    short_sql = sql.replace("\n", " ").strip()
    if len(short_sql) > 120:
        short_sql = short_sql[:120] + "…"

    is_join   = len(tables) > 1
    tbl_str   = ", ".join(f"[cyan]{t}[/cyan]" for t in tables)
    col_str   = "  ".join(col_parts)

    if is_join:
        join_label = " × ".join(f"[bold magenta]{t}[/bold magenta]" for t in tables)
        console.print(f"  [bold magenta]⟂ CROSS-TABLE JOIN:[/bold magenta] {join_label}")
        console.print(f"  [bold magenta]▶ Objective:[/bold magenta] {reason}")
    else:
        console.print(f"  [dim cyan]▶ Objective:[/dim cyan] {reason}")
    console.print(f"    [dim]{'Tables' if is_join else 'Table'}:[/dim] {tbl_str}")
    console.print(f"    [dim]Columns:[/dim] {col_str}")
    console.print(f"    [dim]Executing:[/dim] [dim]{short_sql}[/dim]")
    console.print(f"    [dim]Plain language:[/dim] {plain_language}")
    console.print(f"    [dim]Result:[/dim]")
    for line in result.splitlines()[:5]:
        console.print(f"      [dim]{line}[/dim]")
    extra = len(result.splitlines()) - 5
    if extra > 0:
        console.print(f"      [dim]… ({extra} more lines)[/dim]")
    console.print()


def _print_finish_catalogue(data: dict, accepted: bool) -> None:
    status = "[bold green]ACCEPTED[/bold green]" if accepted else "[bold yellow]REJECTED[/bold yellow]"
    lines = [f"finish_catalogue — {status}\n"]

    desc = data.get("description", "")
    lines.append(f"  [bold]description:[/bold] {desc[:120] + '…' if len(desc) > 120 else desc}")

    tables = data.get("tables", [])
    lines.append(f"  [bold]tables:[/bold] {len(tables)} — {', '.join(t['name'] if isinstance(t, dict) else str(t) for t in tables)}")

    metrics = data.get("measurable_metrics", [])
    lines.append(f"  [bold]measurable_metrics:[/bold] {len(metrics)}")
    for m in metrics:
        name = m.get("name", "?") if isinstance(m, dict) else str(m)
        lines.append(f"    • {name}")

    facts = data.get("queryable_facts", [])
    lines.append(f"  [bold]queryable_facts:[/bold] {len(facts)}")
    for f in facts:
        name = f.get("name", "?") if isinstance(f, dict) else str(f)
        lines.append(f"    • {name}")

    tc = data.get("time_coverage", {})
    if isinstance(tc, dict):
        lines.append(f"  [bold]time_coverage:[/bold] {tc.get('start', '?')} → {tc.get('end', '?')}")

    dqn = data.get("data_quality_notes", [])
    dqn_count = dqn if isinstance(dqn, int) else len(dqn) if isinstance(dqn, list) else 0
    lines.append(f"  [bold]data_quality_notes:[/bold] {dqn_count} note(s)")

    border = "green" if accepted else "yellow"
    console.print(Panel("\n".join(lines), border_style=border, padding=(0, 1)))


def _print_summary(
    catalogue,
    usage: dict,
    elapsed_secs: float,
    metric_results: list,
    fact_results: list,
    uncovered_tables: list[str],
    model: str,
) -> None:
    inp           = usage["input_tokens"]
    out           = usage["output_tokens"]
    cache_created = usage.get("cache_creation_tokens", 0)
    cache_read    = usage.get("cache_read_tokens", 0)
    mins, secs = divmod(int(elapsed_secs), 60)
    elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    cost_str = format_cost(model, inp, out, cache_created, cache_read)

    _STATUS_ICON = {"PASS": "[green]✓[/green]", "WARN": "[yellow]⚠[/yellow]", "FAIL": "[red]✗[/red]"}

    n_pass = sum(1 for r in metric_results if r.status == "PASS")
    n_warn = sum(1 for r in metric_results if r.status == "WARN")
    n_fail = sum(1 for r in metric_results if r.status == "FAIL")

    cross_table_metrics = [m for m in catalogue.measurable_metrics if len(m.tables_used) > 1]
    cross_table_facts   = [f for f in catalogue.queryable_facts   if len(f.tables_used) > 1]

    uncovered_line = (
        f"\n  [yellow]⚠ Uncovered tables:   {', '.join(uncovered_tables)}[/yellow]"
        if uncovered_tables else ""
    )
    header = (
        f"[bold]Catalogue summary[/bold]\n"
        f"  Tables documented:    {len(catalogue.tables)}\n"
        f"  Measurable metrics:   {len(catalogue.measurable_metrics)}  "
        f"([green]{n_pass} pass[/green]  [yellow]{n_warn} warn[/yellow]  [red]{n_fail} fail[/red]"
        + (f"  [bold magenta]⟂ {len(cross_table_metrics)} cross-table[/bold magenta]" if cross_table_metrics else "  [yellow]⚠ 0 cross-table[/yellow]")
        + f")\n"
        f"  Queryable facts:      {len(catalogue.queryable_facts)}"
        + (f"  [dim]([bold magenta]⟂[/bold magenta] {len(cross_table_facts)} cross-table)[/dim]" if cross_table_facts else "")
        + f"\n"
        f"  Time coverage:        {catalogue.time_coverage.start} → {catalogue.time_coverage.end}\n"
        f"  Data quality notes:   {len(catalogue.data_quality_notes)}\n"
        f"  Tokens:               {inp:,} in + {out:,} out"
        + (f" + {cache_created:,} cache write + {cache_read:,} cache read" if cache_created or cache_read else "")
        + f"\n"
        f"  Cost:                 {cost_str}\n"
        f"  Elapsed:              {elapsed_str}\n"
        f"  Model:                {model}"
        f"{uncovered_line}\n"
    )

    metric_lines = "\n".join(
        f"  {_STATUS_ICON.get(r.status, '?')} [{m.confidence}] "
        + (f"[bold magenta]⟂[/bold magenta] " if len(m.tables_used) > 1 else "")
        + f"[cyan]{m.name}[/cyan]"
        + (f"  [dim]({r.error})[/dim]" if r.error else "")
        + f"\n      {m.description}"
        for m, r in zip(catalogue.measurable_metrics, metric_results)
    )

    fact_lines = "\n".join(
        f"  {_STATUS_ICON.get(r.status, '?')} "
        + (f"[bold magenta]⟂[/bold magenta] " if len(f.tables_used) > 1 else "")
        + f"[cyan]{f.name}[/cyan]"
        + (f"  [dim]({r.error})[/dim]" if r.error else "")
        for f, r in zip(catalogue.queryable_facts, fact_results)
    )

    dqn_lines = "\n".join(
        f"  [yellow]•[/yellow] {note}"
        for note in catalogue.data_quality_notes
    )

    body = header
    if metric_lines:
        body += f"\n[bold]Metrics[/bold]\n{metric_lines}\n"
    if fact_lines:
        body += f"\n[bold]Queryable facts[/bold]\n{fact_lines}\n"
    if dqn_lines:
        body += (
            f"\n[bold]Data quality notes[/bold]  "
            f"[dim](caveats a metric consumer must understand to avoid misreading the data: "
            f"null semantics, partial coverage, sparse periods, schema quirks)[/dim]\n"
            f"{dqn_lines}"
        )

    console.print(Panel(body, border_style="green", padding=(0, 1)))


def _render_overview_md(catalogue) -> str:
    """Render a structured Markdown overview of a catalogue (v6 structure)."""
    lines: list[str] = []

    # ── title ─────────────────────────────────────────────────────────────────
    lines.append(f"# {catalogue.description or 'Database Overview'}")

    n_tables  = len(catalogue.tables)
    n_metrics = len(catalogue.measurable_metrics)
    n_facts   = len(catalogue.queryable_facts)
    fact_word = "fact" if n_facts == 1 else "facts"
    date      = catalogue.analysed_at[:10]
    tr_start  = catalogue.time_coverage.start[:7]
    tr_end    = catalogue.time_coverage.end[:7]
    lines.append(
        f"> {date} · {catalogue.model} · {n_tables} tables · "
        f"{n_metrics} metrics · {n_facts} {fact_word} · {tr_start} → {tr_end}"
    )

    # ── overview ─────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("## Overview")
    lines.append(catalogue.overview)

    # ── key terms ────────────────────────────────────────────────────────────
    if catalogue.key_terms:
        lines.append("")
        lines.append("## Key Terms")
        for kt in catalogue.key_terms:
            lines.append(f"- **{kt.term}** — {kt.definition}")

    # ── catalogue-level data quality notes (cross-table / general) ───────────
    if catalogue.data_quality_notes:
        lines.append("")
        lines.append("## Data Quality Notes")
        for i, note in enumerate(catalogue.data_quality_notes, 1):
            lines.append(f"{i}. {note}")

    # ── tables at a glance ───────────────────────────────────────────────────
    lines.append("")
    lines.append("## Tables at a Glance")
    lines.append("")
    lines.append("| Table | Rows | What it holds |")
    lines.append("|---|---:|---|")
    for t in catalogue.tables:
        lines.append(f"| <u>**{t.name}**</u> | {t.row_count:,} | {t.description} |")

    # ── table relationships (mermaid) ─────────────────────────────────────────
    if catalogue.table_relationships:
        lines.append("")
        lines.append("## Table Relationships")
        lines.append("")
        lines.append("```mermaid")
        lines.append('%%{init: {"flowchart": {"curve": "linear"}}}%%')
        lines.append("flowchart TD")
        defined: set[str] = set()

        def _node(name: str) -> str:
            if name not in defined:
                defined.add(name)
                return f'{name}["<u><b>{name}</b></u>"]'
            return name

        for rel in catalogue.table_relationships:
            a = _node(rel.table_a)
            b = _node(rel.table_b)
            lines.append(
                f'    {a} -->|"<u><i><b>{rel.join_key}</b></i></u>"| {b}'
            )
        lines.append("```")
        lines.append("")
        lines.append(
            "> **Legend:** An arrow **A → B** means table A holds the foreign key "
            "column (shown as the arrow label) that references table B. "
            "To join two tables, match the labelled column across both."
        )

    # ── tables reference ──────────────────────────────────────────────────────
    lines.append("")
    lines.append("## Tables Reference")
    for t in catalogue.tables:
        lines.append("")
        lines.append(f"### <u>**{t.name}**</u>")
        lines.append(t.description)
        if t.key_columns:
            cols = ", ".join(f"<u>***{c}***</u>" for c in t.key_columns)
            lines.append(f"Columns: {cols}")
        if t.data_quality_notes:
            lines.append("")
            lines.append("> **Data notes**")
            for note in t.data_quality_notes:
                lines.append(f"> - {note}")

    # ── metrics ───────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("## Metrics")
    lines.append("")

    # Group metrics — use group field if set, else fall back to primary table
    table_set = {t.name for t in catalogue.tables}

    def _primary_table(tables_used: list[str]) -> str:
        for t in tables_used:
            if t in table_set:
                return t
        return "Other"

    def _metric_group(m) -> str:
        g = getattr(m, "group", "") or ""
        return g if g else _primary_table(m.tables_used)

    seen_groups: list[str] = []
    groups: dict[str, list] = {}
    for m in catalogue.measurable_metrics:
        g = _metric_group(m)
        if g not in groups:
            seen_groups.append(g)
            groups[g] = []
        groups[g].append(m)

    if seen_groups:
        group_labels = ", ".join(f"**{g}**" for g in seen_groups)
        lines.append(
            f"> The thematic groups below ({group_labels}) are organisational "
            "labels for this document — they are not tables, columns, or any "
            "named entity in the database."
        )

    for g in seen_groups:
        lines.append("")
        lines.append(f"### {g}")
        lines.append("")
        for m in groups[g]:
            lines.append(f"- **{m.name}**")
            tr = f"{m.time_range.start[:7]} → {m.time_range.end[:7]}"
            lines.append(
                f"  ***Frequency:*** {m.granularity} · "
                f"***Unit:*** {m.unit} · ***Range:*** {tr}"
            )
            lines.append(f"  {m.description}")
            if m.tables_used:
                tables_str = ", ".join(f"<u>**{t}**</u>" for t in m.tables_used)
                lines.append(f"  Tables: {tables_str}")
            lines.append("  ```sql")
            for sql_line in m.sql.split("\n"):
                lines.append(f"  {sql_line}")
            lines.append("  ```")
            if m.agent_notes:
                lines.append(f"  > {m.agent_notes}")
            lines.append("")

    # ── facts ─────────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("## Facts")
    lines.append("")
    lines.append(
        "> Unlike metrics, facts are not time-series. Each query below returns "
        "a current snapshot or a static reference table — a single result set, "
        "not a trend over time."
    )
    lines.append("")
    for f in catalogue.queryable_facts:
        lines.append(f"### {f.name}")
        lines.append(f.description)
        if f.tables_used:
            tables_str = ", ".join(f"<u>**{t}**</u>" for t in f.tables_used)
            lines.append(f"Tables: {tables_str}")
        lines.append("```sql")
        lines.append(f.sql)
        lines.append("```")
        if f.agent_notes:
            lines.append(f"> {f.agent_notes}")
        lines.append("")

    return "\n".join(lines) + "\n"


def _write_output(catalogue, out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(catalogue.model_dump(), f, indent=2)
    console.print(f"\n[bold green]Catalogue written to {out_path}[/bold green]")

    if catalogue.overview:
        # Derive overview path: <stem replacing "_catalogue_N" with "_overview_N">.md
        overview_path = p.parent / p.name.replace("_catalogue_", "_overview_").replace(".json", ".md")
        overview_path.write_text(_render_overview_md(catalogue))
        console.print(f"[bold green]Overview written to {overview_path}[/bold green]")
