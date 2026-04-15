"""
agent.py — Schematica.

Two-phase autonomous loop:

  Phase 1 — Exploration
    The agent has run_query available. It introspects the schema, runs
    validation queries, checks date ranges, confirms aggregation logic.
    Budget scales with database size: min(10 + n_tables * 3, 50) iterations.
    The agent must use at least half the budget (min 10 iters) before finishing.

  Phase 2 — Documentation (enforced)
    At the phase boundary, run_query is removed. The agent has only
    finish_catalogue available and must compile and submit the catalogue.
    It already has all query results in its context window.

This design ensures:
- Simple databases finish quickly and naturally.
- Complex databases get a generous exploration budget.
- The agent always produces output — it cannot loop indefinitely.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from sqlalchemy import text

from schematica.backends import _AnthropicBackend, _LiteLLMBackend
from schematica.db import make_readonly_engine, prompt_readonly_confirmation

from schematica.catalogue import DataCatalogue
from schematica.eval import evaluate_metric, evaluate_fact, _is_evaluator_crash
from schematica.pricing import get_context_window as _context_window
from schematica.introspect import introspect, render_as_text
from schematica.pricing import format_cost

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

console = Console()


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if val is None:
        raise RuntimeError(f"{name} is not set. Add it to .env or the environment.")
    return val


def _optional_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _check_package_versions(
    pandas_version: str | None = None,
    numpy_version:  str | None = None,
) -> None:
    """Raise RuntimeError if pandas + numpy versions are incompatible.

    pandas < 2.2 uses numpy.rec internally in pd.read_sql(). numpy 2.0 removed
    numpy.rec. The combination silently breaks every Phase 3 eval result.

    Arguments allow injection for testing; production call uses installed versions.
    """
    import pandas
    import numpy

    pv = pandas_version or pandas.__version__
    nv = numpy_version  or numpy.__version__

    def _ver(s: str) -> tuple[int, ...]:
        return tuple(int(x) for x in s.split(".")[:2])

    if _ver(pv) < (2, 2) and _ver(nv) >= (2, 0):
        raise RuntimeError(
            f"Incompatible packages: pandas {pv} + numpy {nv}.\n"
            "pandas < 2.2 uses numpy.rec which was removed in numpy 2.0 — "
            "every eval metric will fail with 'No module named numpy.rec'.\n"
            "Fix: uv add \"pandas>=2.2\" \"numpy>=2.0\""
        )


MODEL              = _require_env("SC_MODEL")
MAX_ROWS           = int(_require_env("SC_MAX_ROWS"))
MAX_CHARS          = int(_require_env("SC_MAX_CHARS"))
_BUDGET_BASE       = int(_require_env("SC_BUDGET_BASE"))
_BUDGET_MULTIPLIER = int(_require_env("SC_BUDGET_MULTIPLIER"))
_BUDGET_CAP        = int(_require_env("SC_BUDGET_CAP"))
_MIN_ITER_FLOOR      = int(_require_env("SC_MIN_ITER_FLOOR"))
_MIN_ITER_DIVISOR    = int(_require_env("SC_MIN_ITER_DIVISOR"))
_REFINEMENT_BUDGET   = int(_require_env("SC_REFINEMENT_BUDGET"))
_MAX_OUTPUT_TOKENS   = int(_require_env("SC_MAX_OUTPUT_TOKENS"))
# SC_CACHE=true uses the native Anthropic SDK with prompt caching.
# Caching requires an anthropic/ model — fail loudly if misconfigured.
_CACHE = _optional_env("SC_CACHE", "false").lower() == "true"
if _CACHE and not MODEL.startswith("anthropic/"):
    raise RuntimeError(
        f"SC_CACHE=true requires a model with the 'anthropic/' prefix, got: {MODEL!r}. "
        "Either set SC_CACHE=false or change SC_MODEL to e.g. anthropic/claude-haiku-4-5-20251001"
    )
# The Anthropic SDK expects the bare model name without the provider prefix.
_ANTHROPIC_MODEL = MODEL[len("anthropic/"):] if MODEL.startswith("anthropic/") else MODEL


def _apply_model_override(new_model: str, cache_override: "bool | None" = None) -> None:
    """Update MODEL, _ANTHROPIC_MODEL, and _CACHE after CLI --model / --cache flags.

    cache_override=True  → --cache flag was passed; enable caching
    cache_override=None  → no flag; keep _CACHE as set by SC_CACHE in .env (default false)

    Rules:
      - Non-anthropic model always sets _CACHE=False (caching is Anthropic-only).
        Passing cache_override=True with a non-anthropic model is an error.
      - Anthropic model with cache_override=True overrides .env value.
      - Anthropic model with no flag leaves _CACHE unchanged.
    """
    global MODEL, _ANTHROPIC_MODEL, _CACHE
    if not new_model.startswith("anthropic/"):
        if cache_override is True:
            raise RuntimeError(
                f"--cache requires a model with the 'anthropic/' prefix, got: {new_model!r}. "
                "Prompt caching is only supported by Anthropic models."
            )
        _CACHE = False
    elif cache_override is True:
        _CACHE = True
    MODEL = new_model
    _ANTHROPIC_MODEL = new_model[len("anthropic/"):] if new_model.startswith("anthropic/") else new_model

# Descriptions for warn codes that appear in Phase 3 eval results.
# Only codes that actually appear in a run are shown in the legend.
_PHASE3_WARN_LEGEND: dict[str, str] = {
    "zero_rows":        "SQL ran without error but returned 0 rows — filter condition may be wrong or data is absent",
    "sparse":           "Fewer than 3 rows returned — not enough data points for a reliable metric",
    "high_nulls":       "Value column has >10% NULL entries — may silently skew aggregations",
    "date_mismatch":    "Actual data range falls outside the declared time_range — auto-patched below",
    "extra_cols":       "Query returns more than 2 columns — metrics must return exactly date + value",
    "period_boundary":  "time_range start/end does not align to the granularity boundary (e.g. monthly → first of month) — auto-patched below",
}


def _tables_referenced_in_sql(sql: str) -> set[str]:
    """
    Return the set of table names that appear after FROM or JOIN in a SQL statement.

    Handles:
      - Bare identifiers:          FROM orders
      - Double-quoted identifiers: FROM "Order Details"
      - Bracket-quoted:            FROM [Order Details]
      - Backtick-quoted:           FROM `order_details`
      - Schema-qualified:          FROM public.orders     (returns "orders")
      - Quoted schema+table:       FROM "public"."orders" (returns "orders")
    """
    # An identifier is any of: "...", `...`, [...], or bare word characters.
    _IDENT = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|\w+)'
    # After FROM/JOIN: optionally consume a schema prefix (identifier + dot),
    # then capture the table identifier in groups 1-4 below.
    _TABLE = r'(?:"([^"]+)"|`([^`]+)`|\[([^\]]+)\]|(\w+))'
    pattern = rf'\b(?:FROM|JOIN)\s+(?:{_IDENT}\s*\.\s*)?{_TABLE}'

    tables: set[str] = set()
    for groups in re.findall(pattern, sql, re.IGNORECASE):
        table = next((g for g in groups if g), None)
        if table:
            tables.add(table.lower())
    return tables


def _tables_used_violations(items: list[dict]) -> list[str]:
    """
    Return a list of human-readable errors for any metric or fact where tables_used
    lists a table that does not appear in the SQL's FROM/JOIN clauses.
    """
    violations = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "?")
        sql = item.get("sql", "")
        declared = {t.lower() for t in item.get("tables_used", [])}
        referenced = _tables_referenced_in_sql(sql)
        ghost = declared - referenced
        if ghost:
            violations.append(
                f"'{name}': tables_used lists {sorted(ghost)} but those tables do not "
                f"appear in the SQL FROM/JOIN clauses — remove them or fix the SQL to actually join them"
            )
    return violations


def _uncovered_fk_pairs(
    metrics: list[dict],
    fk_pairs: list[tuple[str, str]],
    lookup_tables: set[str] | None = None,
) -> list[tuple[str, str]]:
    """
    Return FK pairs that have no covering metric.

    A FK pair (table_a, table_b) is considered covered when at least one metric's
    SQL references both tables in its FROM/JOIN clauses.  Direction is ignored:
    (orders, customers) is covered whether the SQL says FROM orders JOIN customers
    or FROM customers JOIN orders.

    Parameters
    ----------
    metrics : list[dict]
        Submitted measurable_metrics (facts are excluded — they don't count).
    fk_pairs : list[tuple[str, str]]
        Each tuple is (from_table, to_table) as returned by the schema snapshot.
    """
    if not fk_pairs:
        return []

    exempt = {t.lower() for t in (lookup_tables or set())}
    uncovered = []
    for from_table, to_table in fk_pairs:
        ft, tt = from_table.lower(), to_table.lower()
        if ft in exempt or tt in exempt:
            continue
        pair = frozenset({ft, tt})
        covered = any(
            pair <= _tables_referenced_in_sql(m.get("sql", ""))
            for m in metrics
            if isinstance(m, dict)
        )
        if not covered:
            uncovered.append((ft, tt))
    return uncovered


def _phase1_budget(n_tables: int) -> int:
    """Exploration iteration budget, scales with database size."""
    return min(_BUDGET_BASE + n_tables * _BUDGET_MULTIPLIER, _BUDGET_CAP)


_JSON_STRING_FIELDS: dict[str, type] = {
    "tables": list,
    "measurable_metrics": list,
    "queryable_facts": list,
    "time_coverage": dict,
}


def _coerce_json_strings(data: dict) -> dict:
    """Return a copy of *data* with JSON-encoded string fields parsed to native types.

    Some LLMs double-encode nested structures — e.g. submitting
    ``"tables": "[{...}]"`` instead of ``"tables": [{...}]``.
    For each known list/dict field, if the value is a string we attempt
    ``json.loads``; on failure the original string is kept so downstream
    validators can produce a meaningful error.
    """
    out = dict(data)
    for field, expected_type in _JSON_STRING_FIELDS.items():
        val = out.get(field)
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
            except (ValueError, TypeError):
                pass
            else:
                if isinstance(parsed, expected_type):
                    out[field] = parsed
    return out


def _format_iter_stats(
    in_tokens: int,
    out_tokens: int,
    model: str,
    pricing: dict | None = None,
    tracker: "_RequestTracker | None" = None,
    now: float = 0.0,
    iter_duration: float = 0.0,
    total_in: int = 0,
    total_out: int = 0,
    total_cost: float = 0.0,
    iter_num: int = 0,
    max_iter: int = 0,
    context_window: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> str:
    """Return a box-formatted stats block for the ↳ separator between iterations.

    The box has two rows:
      call  — per-iteration tokens, cost, and wall-clock duration
      total — accumulated tokens, cost, elapsed time, req count, rpm, this-min
              (only shown when a tracker is provided)
    """
    from schematica.pricing import CACHE_WRITE_MULTIPLIER, CACHE_READ_MULTIPLIER, get_model_pricing

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

    _iter_label = f"current iter {iter_num}/{max_iter}" if iter_num and max_iter else "current iter"
    _ITER_HDR  = f"─ {_iter_label} "
    _ACCUM_HDR = "─ accumulated "
    _FOOTER    = "─ each iteration = 1 LLM call "
    _SEP       = " · "

    # Total effective input for context-fill % includes all token types
    effective_in = in_tokens + cache_creation_tokens + cache_read_tokens
    _fill = f"{_SEP}{effective_in / context_window * 100:.1f}% ctx" if context_window > 0 else ""

    # Build optional cache fields shown inline after "in"
    _cache_parts = ""
    if cache_read_tokens:
        _cache_parts += f"{_SEP}{cache_read_tokens:,} cached"
    if cache_creation_tokens:
        _cache_parts += f"{_SEP}{cache_creation_tokens:,} cache↑"

    iter_content = (
        f"  {in_tokens:,} in{_cache_parts}{_SEP}{out_tokens:,} out{_SEP}${iter_cost:.4f}{_SEP}{_fmt_dur(iter_duration)}{_fill}  "
    )

    if tracker is not None and tracker.total > 0:
        elapsed = now - tracker._started_at
        iter_per_min = tracker.rpm(now)
        accum_content = (
            f"  {total_in:,} in{_SEP}{total_out:,} out{_SEP}${total_cost:.4f}"
            f"{_SEP}{_fmt_dur(elapsed)}{_SEP}{iter_per_min:.1f} iter/min  "
        )
        inner_w = max(
            len(iter_content),
            len(accum_content),
            len(_ITER_HDR) + 2,
            len(_ACCUM_HDR) + 2,
            len(_FOOTER) + 2,
        )
        iter_padded  = iter_content.ljust(inner_w)
        accum_padded = accum_content.ljust(inner_w)
        top    = "╭" + _ITER_HDR  + "─" * (inner_w - len(_ITER_HDR))  + "╮"
        mid    = "├" + _ACCUM_HDR + "─" * (inner_w - len(_ACCUM_HDR)) + "┤"
        bottom = "╰" + _FOOTER    + "─" * (inner_w - len(_FOOTER))    + "╯"
        return "\n".join([top, f"│{iter_padded}│", mid, f"│{accum_padded}│", bottom])
    else:
        inner_w = max(len(iter_content), len(_ITER_HDR) + 2, len(_FOOTER) + 2)
        iter_padded = iter_content.ljust(inner_w)
        top    = "╭" + _ITER_HDR + "─" * (inner_w - len(_ITER_HDR)) + "╮"
        bottom = "╰" + _FOOTER   + "─" * (inner_w - len(_FOOTER))   + "╯"
        return "\n".join([top, f"│{iter_padded}│", bottom])


def _calc_rpm(n_requests: int, elapsed_secs: float) -> float:
    """Return requests-per-minute rate; 0.0 when there is no data yet."""
    if n_requests <= 0 or elapsed_secs <= 0:
        return 0.0
    return (n_requests / elapsed_secs) * 60.0


def _retry_after_seconds(exc: Exception) -> float | None:
    """Return the retry-after hint from a rate-limit exception header, or None.

    Works for any backend that exposes exc.response.headers['retry-after'].
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", {}) or {}
    val = headers.get("retry-after")
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# Conservative default for Anthropic tier-1 accounts.  Updated at runtime from
# the anthropic-ratelimit-output-tokens-limit response header.
_DEFAULT_OUTPUT_TPM = 16_000


class _OutputTokenBucket:
    """Rolling 60-second window tracker for proactive output-token rate limiting.

    Only used with the Anthropic backend, where the per-minute limit can be read
    from response headers.  Records actual output tokens after each API call and
    sleeps before the next call when the expected output would exceed the limit.
    """

    WINDOW = 60.0  # seconds

    def __init__(self, limit: int = _DEFAULT_OUTPUT_TPM) -> None:
        self._limit = limit
        self._window: deque[tuple[float, int]] = deque()

    def update_limit(self, limit: int) -> None:
        """Update the per-minute output token limit (from API response headers)."""
        self._limit = limit

    def record(self, now: float, tokens: int) -> None:
        """Record that `tokens` output tokens were generated at wall-clock `now`."""
        self._evict(now)
        if tokens > 0:
            self._window.append((now, tokens))

    def tokens_in_window(self, now: float) -> int:
        """Sum of output tokens generated in the last 60 seconds."""
        self._evict(now)
        return sum(t for _, t in self._window)

    def _evict(self, now: float) -> None:
        while self._window and now - self._window[0][0] >= self.WINDOW:
            self._window.popleft()

    def proactive_wait(self, now: float, expected: int) -> float:
        """Sleep until there is headroom for `expected` output tokens.

        Iterates the window oldest-first, dropping entries until used + expected
        fits within the limit.  The wait duration is the time until the newest
        entry that must be dropped leaves the 60-second window.
        Returns seconds slept (0.0 if no wait was needed).
        """
        self._evict(now)
        used = sum(t for _, t in self._window)
        if used + expected <= self._limit:
            return 0.0

        need_to_drop = used + expected - self._limit
        dropped = 0
        wait_secs = 0.0
        for ts, tok in self._window:  # oldest → newest
            dropped += tok
            age = now - ts
            wait_secs = self.WINDOW - age + 0.5  # 0.5 s safety buffer
            if dropped >= need_to_drop:
                break

        if wait_secs > 0:
            console.print(
                f"[yellow]  Output token budget ({used:,}/{self._limit:,} per min) — "
                f"waiting {wait_secs:.0f}s before next call[/yellow]"
            )
            time.sleep(wait_secs)
        return wait_secs


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


SYSTEM_PROMPT = """\
You are the Schematica. Your job is to analyse a database and produce \
a precise, structured catalogue describing what data is available and how it can \
be measured as time-series metrics.

You work in two phases:

THE SCHEMA SNAPSHOT IS GROUND TRUTH
The schema snapshot you received was computed against the full dataset before this \
session started. Its statistics are exact, not sampled:
- Column min/max values are the true boundaries of the data — use these directly \
  for time_range.start and time_range.end in every metric.
- top_values for categorical columns are exhaustive counts — use them to identify \
  distinct values (e.g. revenue types, event types, outcomes) without querying.
- n_null is the exact null count for every column — use it for data_quality_notes \
  and confidence assessment without querying. data_quality_notes must never be \
  empty for a real database. Include anything a metric consumer would need to know \
  to avoid misreading a metric: high null rates (especially where null means \
  "not applicable" or "not yet recorded" rather than zero), partial coverage \
  (e.g. a table that only contains rows matching a specific status or tier), \
  date/time gaps or sparse periods, and any schema quirk that silently skews \
  aggregation results.
- row_count per table is exact.
- Foreign keys are declared — use them for join planning without querying.

Only use run_query to validate aggregation logic and SQL correctness. Never use it \
to rediscover facts the snapshot already provides. If a query result appears to \
contradict the snapshot, trust the snapshot.

PHASE 1 — EXPLORATION
You have the run_query tool. Use it to:
- Validate that your proposed SQL queries actually return correct results
- Confirm that GROUP BY / aggregation logic works as expected
- Check that joins between tables produce the expected shape

Run queries efficiently — batch your validation where possible. You do not need \
to validate every possible metric variant, just confirm that your core queries work.

PHASE 2 — DOCUMENTATION
When you are told exploration is complete, you will have only finish_catalogue \
available. Compile everything you have learned and submit the catalogue. \
You already have all the information you need from Phase 1.

General rules:
- The "overview" field is a multi-paragraph narrative (3–6 paragraphs, plain prose, \
  no bullet points) written for someone who has never seen this database before. It must cover: \
  (1) what real-world domain or business this database serves; \
  (2) what each table represents and what entity or event it captures; \
  (3) key relationships between tables (who links to whom and why); \
  (4) what kinds of analysis or questions the data supports; \
  (5) any important context about data coverage, history, or limitations. \
  The "description" field is a single sentence summary of the same.
- Prefer monthly aggregations for event-level data.
- Only include SQL that you have validated by running it in Phase 1.
- Be honest about confidence: high = unambiguous columns, medium = inferred \
  from naming/samples, low = significant uncertainty.
- Do not hallucinate column names — only use columns that appear verbatim in the \
  schema snapshot. If a query returns a column error, look up the exact column name \
  in the schema snapshot and fix the SQL before retrying. Never guess.
- Cover all tables in the tables summary, even those with no measurable metrics.
- Aim for quality over quantity: target 3–5 metrics per table, up to ~70 metrics \
  total for large databases. A complex database does not need every possible \
  permutation — prefer the most insightful and distinct metrics over exhaustive \
  coverage. If many breakdowns of the same base metric are possible, pick the most \
  useful 2–3; do not generate one metric per categorical value.
- time_range start and end must align to the metric's granularity boundary: \
  for monthly metrics both dates must be the first day of a month; for annual \
  metrics the first day of a year. Use the snapshot column min/max and truncate \
  to the nearest period boundary — never use a mid-period date as a range endpoint. \
  For "tick" granularity (un-aggregated, one row per raw event/record) any ISO date \
  is acceptable — there is no period boundary to align to.
- When two metrics measure a similar concept from different source tables, \
  document in agent_notes how they differ and when to prefer each.
- For every quantifiable entity, consider both forms: a flow metric (how many \
  per period) and a stock metric (running total or cumulative count at the end of \
  each period — one row per period, not a single current snapshot). Include both \
  where the data supports it. A stock metric is ALWAYS a time series: it must \
  return one row per period with (period_date, cumulative_value). A query that \
  returns a single scalar (e.g. "SELECT COUNT(*) FROM accounts WHERE active") is \
  NOT a metric — it is a queryable fact (point-in-time snapshot). Do not place \
  single-scalar queries in measurable_metrics.
- Where a count metric is meaningful, consider whether the rate or percentage \
  form (count ÷ relevant base) is also worth including as a separate metric.
- When breaking a metric down by a categorical column, cover ALL distinct values \
  shown in top_values in the snapshot — do not silently omit any.
- When SQL filters on a coded value, explain all codes used in agent_notes \
  so consumers understand what is and is not included.
- Every table must appear in at least one metric OR one queryable fact. A table \
  with no metrics should be preserved as a queryable fact unless it is empty or \
  its data is fully superseded by another table — in which case explain this in \
  data_quality_notes.
- CROSS-TABLE METRICS ARE REQUIRED when foreign keys exist. For every foreign key \
  relationship in the schema snapshot, consider what derived time-series metric the \
  join enables: a rate (events per entity per period), a ratio (value from one table \
  divided by a count or quantity from another, grouped by period), or a combined \
  aggregate that is only meaningful with both tables. Include at least one join-based \
  time-series metric per foreign key relationship where the join produces an \
  interpretable result. A queryable fact that uses a join does NOT satisfy this \
  requirement — the metric must return a time column and a single aggregated value \
  over time. If no useful time-series metric is possible from a join, explain why in \
  data_quality_notes — do not silently skip cross-table analysis.
- When exploring a foreign key join in Phase 1, always run at least one query that \
  groups by a period column (e.g. GROUP BY period_month, value) to validate the \
  time-series form of the join — not just a static aggregate. If the time-series \
  query returns a valid (period, value) result, that join must produce a metric, \
  not a fact.
- Each TableSummary must include a data_quality_notes list with observations specific \
  to that table: high null counts (with exact counts from n_null), sparse columns, \
  ambiguous values, or quirks that would affect metrics built on that table. \
  Move per-table observations here rather than to the catalogue-level data_quality_notes. \
  Reserve catalogue-level data_quality_notes for cross-table observations and general \
  database limitations only.
- Every metric must have a group field: a short thematic label (2-4 words) shared \
  with related metrics, e.g. 'Revenue', 'Customer Accounts', 'Product Usage', \
  'Support & Escalations'. Use 3-6 groups that reflect the natural analytical \
  domains in the data. All metrics in the same analytical area must share the same \
  group string exactly.
- key_terms: Identify 5-10 domain-specific terms that appear in metric names, \
  descriptions, or column names that a reader unfamiliar with this business domain \
  would need defined. Each term needs a plain-English definition (1-2 sentences). \
  Include only business/domain vocabulary specific to this data — not generic \
  technical terms like SQL or database.
- table_relationships: For each foreign key relationship declared in the schema \
  snapshot, record table_a (the table that holds the FK column), table_b (the \
  table being referenced), and join_key (the shared column name used to join them).

METRIC SQL MUST RETURN EXACTLY 2 COLUMNS: a date/period column and a single \
numeric value column. Never include a category or dimension column as a third column.

If a breakdown by category is meaningful, create one metric per category using \
a WHERE clause to filter to that value — do not group by the category column. \
If only the overall total matters, aggregate across all categories. \
If the full breakdown table is useful to preserve, put it in queryable_facts \
instead — queryable_facts have no column constraints.

QUERYABLE FACTS
Beyond time-series metrics, look for data that is worth preserving but is not a \
time series. Document these as queryable_facts:
- Reference / lookup tables (regions, product categories, status codes, etc.)
- Static or slowly-changing dimension tables
- Point-in-time snapshots (current totals, active record counts)
- Any table whose primary value is its current state rather than a trend over time
- If a fact reflects state at catalogue generation time, say so in agent_notes \
  so consumers know it is not a live feed.
- Do not put coverage statistics or schema observations in queryable_facts — \
  those belong in data_quality_notes.

For each queryable fact, write a SQL query that fetches the relevant data. These \
queries have no column or shape constraints — return whatever columns are useful. \
Validate the SQL in Phase 1 before including it.
"""


REFINEMENT_SYSTEM_PROMPT = """\
You are the Schematica in PHASE 3 — REFINEMENT.

The catalogue has been generated and evaluated against the live database. \
Some metrics have issues that need fixing. Your job is to investigate each \
issue using run_query and submit a corrected catalogue via finish_catalogue.

THE SCHEMA SNAPSHOT IS GROUND TRUTH — same rules as before. Column min/max, \
top_values, n_null and row_counts are exact. Trust them over query results.

For each issue you are given:
- zero_rows: The SQL ran but returned 0 rows. Investigate why. If the filter \
  condition is wrong (e.g. wrong value, wrong column), fix the SQL. If the data \
  genuinely has no records matching the filter, remove the metric entirely and \
  note it in data_quality_notes.
- date_mismatch: Already patched automatically — time_range has been corrected. \
  No action needed.
- high_nulls: Fix the JOIN or WHERE clause — a high null rate usually means a \
  missing join condition or a filter that excludes most rows.
- sparse: Too few rows returned. Check if the filter is too restrictive or the \
  data is genuinely sparse — if sparse, note it in data_quality_notes and keep.
- extra_cols: The metric SQL returns more than 2 columns. This is almost always \
  caused by including a category or dimension column alongside the date and value. \
  Choose the fix based on the number of distinct values in the extra column: \
  (a) 1–5 distinct values: split into one metric per value using a WHERE clause. \
    Example: SELECT period, region, COUNT(*) → one metric per region value. \
  (b) 6 or more distinct values: do NOT split (too many metrics). Instead, \
    move the metric to queryable_facts — it is a dimensional breakdown, not a \
    scalar metric. Remove it from measurable_metrics. \
  Never collapse the dimension into a grand-total metric — if an aggregate was \
  wanted it would already exist as a separate metric.
- fewer than 2 columns returned: The metric SQL returns a single scalar with no \
  date column. This is a point-in-time snapshot, not a time-series. Move it to \
  queryable_facts and remove it from measurable_metrics. Do NOT wrap it in a \
  dummy date — that produces fake data.

Submit the COMPLETE corrected catalogue via finish_catalogue — include all \
metrics (fixed and unchanged), not just the ones you fixed. Remove only metrics \
that genuinely have no data.
"""


# ── tool definitions ───────────────────────────────────────────────────────────

_RUN_QUERY_TOOL = {
    "name": "run_query",
    "description": (
        f"Execute a read-only SQL SELECT query. Results truncated to {MAX_ROWS} rows. "
        "Use this to validate column contents, test proposed SQL, check date ranges, "
        "confirm NULL distributions, or verify aggregation logic."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "A valid SQL SELECT statement for the target database dialect.",
            },
            "reason": {
                "type": "string",
                "description": "One sentence: why you are running this query (the objective).",
            },
            "tables": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Table name(s) this query reads from.",
            },
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Column names used in SELECT, WHERE, GROUP BY, or JOIN.",
            },
            "plain_language": {
                "type": "string",
                "description": "The query described in plain English, e.g. 'Count leads grouped by creation month'.",
            },
        },
        "required": ["sql", "reason", "tables", "columns", "plain_language"],
    },
}

_BARE_TABLES_ERROR_MSG = (
    'tables entries must be full objects, not bare strings. '
    'Each entry must look like: '
    '{"name": "orders", "row_count": 12500, "description": "One row per order.", '
    '"key_columns": ["order_id", "created_at"]}. '
    'Do not pass table names as plain strings.'
)

_TABLES_NOT_LIST_ERROR_MSG = (
    'tables must be a list of table objects, not a string. '
    'You submitted tables as a single string value instead of an array. '
    'Pass tables as a JSON array: '
    '[{"name": "orders", "row_count": 12500, "description": "...", "key_columns": [...]}]. '
    'Do not JSON-encode the array — pass the list directly.'
)

_FK_REJECTION_MSG = (
    "Missing cross-table metrics for FK relationships: {pairs_str}. "
    "Look at the run_query calls you already made in this session — if any contain "
    "JOIN clauses between these tables, extract them as measurable_metrics now. "
    "Each FK relationship requires at least one measurable_metric whose SQL JOINs "
    "both tables and returns (period, aggregate_value). "
    "A queryable_fact with a JOIN does not satisfy this requirement."
)

# Number of consecutive FK-only rejections before a pair is waived.
# Prevents infinite rejection loops when the agent cannot or will not produce
# a JOIN metric for a given pair.
_FK_REJECTION_CAP = 2


def _update_fk_waived(
    missing_fks: list[tuple[str, str]],
    fk_rejection_counts: dict,
    fk_waived: set,
    cap: int = _FK_REJECTION_CAP,
) -> tuple[dict, set]:
    """Increment per-pair rejection counts; move any pair that hits *cap* into *fk_waived*.

    Returns updated (fk_rejection_counts, fk_waived). Both arguments are mutated
    in-place and also returned for convenience.
    Pair direction is normalised via frozenset so (a, b) and (b, a) share one counter.
    """
    for a, b in missing_fks:
        key = frozenset({a, b})
        fk_rejection_counts[key] = fk_rejection_counts.get(key, 0) + 1
        if fk_rejection_counts[key] >= cap:
            fk_waived.add(key)
    return fk_rejection_counts, fk_waived


_FINISH_CATALOGUE_TOOL = {
    "name": "finish_catalogue",
    "description": (
        "Submit the completed data catalogue. "
        "In Phase 1, call this if you are already satisfied. "
        "In Phase 2, you must call this — it is your only available tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tables": {
                "type": "array",
                "description": (
                    'Each entry is a full object — NOT a string. '
                    'Example: {"name": "orders", "row_count": 12500, '
                    '"description": "One row per order.", '
                    '"key_columns": ["order_id", "created_at"]}.'
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "name":        {"type": "string"},
                        "row_count":   {"type": "integer"},
                        "description": {"type": "string"},
                        "key_columns": {"type": "array", "items": {"type": "string"}},
                        "data_quality_notes": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "row_count", "description", "key_columns"],
                },
            },
            "measurable_metrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":        {"type": "string"},
                        "description": {"type": "string"},
                        "sql":         {"type": "string"},
                        "time_range": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "string"},
                                "end":   {"type": "string"},
                            },
                            "required": ["start", "end"],
                        },
                        "granularity": {
                            "type": "string",
                            "enum": ["daily", "weekly", "monthly", "quarterly", "annual", "tick"],
                        },
                        "unit":        {"type": "string"},
                        "tables_used": {"type": "array", "items": {"type": "string"}},
                        "confidence":  {"type": "string", "enum": ["high", "medium", "low"]},
                        "agent_notes": {"type": "string"},
                        "group": {"type": "string"},
                    },
                    "required": [
                        "name", "description", "sql", "time_range",
                        "granularity", "unit", "tables_used", "confidence", "agent_notes",
                        "group",
                    ],
                },
            },
            "time_coverage": {
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "end":   {"type": "string"},
                },
                "required": ["start", "end"],
            },
            "queryable_facts": {
                "type": "array",
                "description": "Non-time-series data worth preserving: reference tables, static lookups, point-in-time snapshots.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":        {"type": "string"},
                        "description": {"type": "string"},
                        "sql":         {"type": "string"},
                        "tables_used": {"type": "array", "items": {"type": "string"}},
                        "agent_notes": {"type": "string"},
                    },
                    "required": ["name", "description", "sql", "tables_used", "agent_notes"],
                },
            },
            "data_quality_notes": {
                "type": "array",
                "items": {"type": "string"},
            },
            "key_terms": {
                "type": "array",
                "description": "Domain-specific terms and their plain-English definitions",
                "items": {
                    "type": "object",
                    "properties": {
                        "term":       {"type": "string"},
                        "definition": {"type": "string"},
                    },
                    "required": ["term", "definition"],
                },
            },
            "table_relationships": {
                "type": "array",
                "description": "Foreign key relationships between tables",
                "items": {
                    "type": "object",
                    "properties": {
                        "table_a":  {"type": "string", "description": "Table holding the FK"},
                        "table_b":  {"type": "string", "description": "Referenced table"},
                        "join_key": {"type": "string", "description": "Shared column name"},
                    },
                    "required": ["table_a", "table_b", "join_key"],
                },
            },
            "description": {
                "type": "string",
                "description": (
                    "A short, title-worthy name for this database (3–6 words). "
                    "Use title case. Examples: 'SaaS Business Database', "
                    "'E-Commerce Orders Database', 'Music Streaming Database'. "
                    "Do NOT write a full sentence — this is used as a document title."
                ),
            },
            "overview": {
                "type": "string",
                "description": (
                    "Multi-paragraph narrative for someone unfamiliar with this database. "
                    "Cover: what real-world domain it serves, what each table represents, "
                    "key relationships between tables, what kinds of analysis the data supports, "
                    "and any important context about data coverage or history. "
                    "3-6 paragraphs. Plain prose — no bullet points."
                ),
            },
        },
        "required": [
            "tables", "measurable_metrics", "queryable_facts",
            "time_coverage", "data_quality_notes",
            "description", "overview",
            "key_terms", "table_relationships",
        ],
    },
}


# ── main entry point ───────────────────────────────────────────────────────────

def run(connection_string: str, out_path: str) -> DataCatalogue:
    """
    Run the Schematica against a database.

    Introspects the schema, runs a two-phase agentic loop to identify and
    validate measurable metrics, then writes the DataCatalogue to out_path.
    """
    _check_package_versions()
    _print_header(connection_string, out_path)

    # For SQLite, verify the file exists before spending any tokens.
    if connection_string.startswith("sqlite:///"):
        db_file = connection_string[len("sqlite:///"):]
        if not os.path.exists(db_file):
            console.print(f"[bold red]Error:[/bold red] SQLite database not found: {db_file}")
            raise SystemExit(1)

    _probe_connection(make_readonly_engine(connection_string), connection_string)

    console.print("[dim]Introspecting schema…[/dim]")
    snapshot = introspect(connection_string)
    schema_text = render_as_text(snapshot)
    _print_schema_summary(snapshot)
    _print_schema_detail(schema_text)

    table_columns: dict[str, list[str]] = {
        t["name"]: [c["name"] for c in t["columns"]]
        for t in snapshot["tables"]
    }

    # Collect all FK pairs so _agent_loop can validate that the catalogue
    # contains at least one cross-table metric per FK relationship.
    fk_pairs: list[tuple[str, str]] = [
        (t["name"], fk["to_table"])
        for t in snapshot["tables"]
        for fk in t.get("foreign_keys", [])
    ]

    # Small tables (≤20 rows) are lookup/reference tables used to enrich other
    # metrics — they don't need their own cross-table time-series metric.
    _LOOKUP_ROW_THRESHOLD = 20

    # Pure junction tables (every column is a FK column, e.g. PlaylistTrack with
    # only PlaylistId and TrackId) have no temporal data and cannot produce a
    # time-series metric on their own — exempt them too.
    _fk_cols_by_table: dict[str, set[str]] = {
        t["name"].lower(): {
            col.lower()
            for fk in t.get("foreign_keys", [])
            for col in fk.get("from_cols", [])
        }
        for t in snapshot["tables"]
    }
    junction_tables: set[str] = {
        t["name"].lower()
        for t in snapshot["tables"]
        if (cols := {c["name"].lower() for c in t.get("columns", [])})
        and cols == _fk_cols_by_table.get(t["name"].lower(), set())
    }

    lookup_tables: set[str] = {
        t["name"].lower()
        for t in snapshot["tables"]
        if t.get("row_count", 0) <= _LOOKUP_ROW_THRESHOLD
    } | junction_tables

    engine = make_readonly_engine(connection_string)
    n_tables = len(snapshot["tables"])
    budget = _phase1_budget(n_tables)
    min_iter = max(_MIN_ITER_FLOOR, budget // _MIN_ITER_DIVISOR)
    console.print(f"[dim]Exploration budget: {budget} iterations for {n_tables} tables  (min {min_iter} before finish)[/dim]\n")

    usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_tokens": 0, "cache_read_tokens": 0, "total_cost": 0.0}
    started_at = time.monotonic()
    req_tracker = _RequestTracker(started_at)
    try:
        catalogue_data = _agent_loop(schema_text, engine, budget, min_iter, usage, table_columns, fk_pairs, lookup_tables, started_at, req_tracker)
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        catalogue_data = None
    elapsed_secs = time.monotonic() - started_at

    if catalogue_data is None:
        inp           = usage["input_tokens"]
        out           = usage["output_tokens"]
        cache_created = usage.get("cache_creation_tokens", 0)
        cache_read    = usage.get("cache_read_tokens", 0)
        mins, secs = divmod(int(elapsed_secs), 60)
        elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        console.print(
            f"[bold red]Agent failed to submit a valid catalogue after exhausting all retries.[/bold red]\n"
            "The agent either ran out of budget or repeatedly submitted incomplete finish_catalogue calls.\n"
            "No output file written. Re-run with a larger budget or inspect the agent log above.\n"
            f"[dim]Tokens: {inp:,} in + {out:,} out"
            + (f" + {cache_created:,} cache write + {cache_read:,} cache read" if cache_created or cache_read else "")
            + f"  |  Cost: {format_cost(MODEL, inp, out, cache_created, cache_read)}"
            + f"  |  Elapsed: {elapsed_str}[/dim]"
        )
        raise SystemExit(1)

    catalogue = _build_catalogue(catalogue_data, snapshot)
    catalogue = _drop_broken_sql(catalogue, engine)
    catalogue, final_metric_results, final_fact_results, uncovered_tables = _run_phase3_safe(
        _run_phase3, catalogue, schema_text, engine, usage, table_columns, tracker=req_tracker
    )
    _write_output(catalogue, out_path)
    _print_summary(catalogue, usage, elapsed_secs, final_metric_results, final_fact_results, uncovered_tables)

    return catalogue


def _call_with_retry(backend, tools: list, max_tokens: int = _MAX_OUTPUT_TOKENS, max_attempts: int = 7):
    """Call backend.call() with backoff on rate-limit errors.

    Uses the retry-after header hint from the exception when available (exact
    wait the API requested).  Falls back to exponential backoff otherwise.
    """
    delay = 30
    for attempt in range(1, max_attempts + 1):
        try:
            return backend.call(tools, max_tokens)
        except Exception as exc:
            msg = str(exc).lower()
            is_unsupported_tools = (
                "does not support parameters" in msg and "tools" in msg
            ) or (
                "unsupportedparams" in msg and "tools" in msg
            )
            if is_unsupported_tools:
                raise RuntimeError(
                    f"Model does not support tool calls, which schematica requires. "
                    "Choose a model with function/tool calling support "
                    "(e.g. gemini/gemini-2.5-flash, anthropic/claude-sonnet-4-6)."
                ) from exc
            is_rate_limit = "rate limit" in msg or "ratelimit" in msg or "429" in msg or "rate_limited" in msg
            is_transient = "empty choices" in msg or "overloaded" in msg or "503" in msg or "502" in msg
            if not (is_rate_limit or is_transient) or attempt == max_attempts:
                raise
            hint = _retry_after_seconds(exc)
            wait = hint if hint is not None else delay
            source = "API hint" if hint is not None else "own backoff"
            console.print(
                f"[yellow]  Rate limit hit — waiting {wait:.0f}s before retry "
                f"(attempt {attempt}/{max_attempts - 1}, {source})[/yellow]"
            )
            time.sleep(wait)
            if hint is None:
                delay = min(delay * 2, 300)


def _make_backend(initial_user_text: str, system_prompt: str) -> "_AnthropicBackend | _LiteLLMBackend":
    """Create the right backend for the configured provider."""
    if _CACHE:
        client = _get_anthropic_client()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": initial_user_text, "cache_control": {"type": "ephemeral"}}
        ]}]
        return _AnthropicBackend(client, _ANTHROPIC_MODEL, system_prompt, messages)
    else:
        messages = [{"role": "user", "content": initial_user_text}]
        return _LiteLLMBackend(MODEL, system_prompt, messages)


# ── agent loop ─────────────────────────────────────────────────────────────────

def _agent_loop(schema_text: str, engine, phase1_budget: int, phase1_min_iter: int, usage: dict, table_columns: dict, fk_pairs: list[tuple[str, str]] | None = None, lookup_tables: set[str] | None = None, started_at: float = 0.0, tracker: "_RequestTracker | None" = None) -> dict:
    initial_message = (
        f"Here is the complete schema snapshot of the database:\n\n"
        f"```\n{schema_text}\n```\n\n"
        f"You are in PHASE 1 — EXPLORATION. You have {phase1_budget} iterations "
        f"to explore the schema and validate your SQL queries. "
        f"Start by identifying what each table represents, then run queries to "
        f"validate the metrics you plan to include in the catalogue."
    )

    # Both Phase 1 and Phase 2 share the same backend — the message history
    # from Phase 1 carries forward into Phase 2.
    backend = _make_backend(initial_message, SYSTEM_PROMPT)

    # Phase 1 — exploration with both tools available.
    catalogue_data, last_rejection_reasons, _ = _run_phase(
        backend, engine,
        tools=[_RUN_QUERY_TOOL, _FINISH_CATALOGUE_TOOL],
        max_iter=phase1_budget,
        min_iter=phase1_min_iter,
        phase_label="1 (exploration)",
        usage=usage,
        table_columns=table_columns,
        fk_pairs=fk_pairs,
        lookup_tables=lookup_tables,
        tracker=tracker,
    )
    if catalogue_data is not None:
        return catalogue_data

    # Phase 2 — documentation, only finish_catalogue available
    console.print(Panel(
        "[bold cyan]Phase 1 complete — entering Phase 2 (documentation)[/bold cyan]\n"
        "[dim]run_query is no longer available. The agent must now compile and submit finish_catalogue.[/dim]",
        border_style="cyan",
        padding=(0, 1),
    ))

    if last_rejection_reasons:
        reasons_text = "; ".join(last_rejection_reasons)
        phase2_prompt = (
            "PHASE 1 is now complete. You are entering PHASE 2 — DOCUMENTATION. "
            "The run_query tool is no longer available. "
            f"Your last finish_catalogue submission was rejected: {reasons_text}. "
            "Fix these issues and resubmit finish_catalogue now with every required field "
            "populated with the data you discovered in Phase 1."
        )
    else:
        phase2_prompt = (
            "PHASE 1 is now complete. You are entering PHASE 2 — DOCUMENTATION. "
            "The run_query tool is no longer available. "
            "You have all the query results you need in this conversation. "
            "Compile everything you have learned and call finish_catalogue now."
        )

    backend.append_user(phase2_prompt)

    catalogue_data, _, _ = _run_phase(
        backend, engine,
        tools=[_FINISH_CATALOGUE_TOOL],
        max_iter=5,
        phase_label="2 (documentation)",
        usage=usage,
        table_columns=table_columns,
        tracker=tracker,
    )
    if catalogue_data is not None:
        return catalogue_data

    raise RuntimeError("Agent did not produce a catalogue after both phases.")


def _run_phase(backend, engine, tools: list, max_iter: int, phase_label: str, usage: dict, table_columns: dict, min_iter: int = 0, fk_pairs: list[tuple[str, str]] | None = None, lookup_tables: set[str] | None = None, tracker: "_RequestTracker | None" = None) -> tuple[dict | None, list, int]:
    """Run one phase of the agent loop. Returns (catalogue_data, last_rejection_reasons, iterations_run)."""
    from schematica.pricing import get_model_pricing

    # Proactive output-token throttling: only for the Anthropic backend, where
    # the per-minute limit is available from response headers.
    output_bucket: _OutputTokenBucket | None = (
        _OutputTokenBucket() if isinstance(backend, _AnthropicBackend) else None
    )
    last_out_tokens: int = 0  # previous iteration's output; used as estimate

    last_rejection_reasons: list[str] = []
    fk_rejection_counts: dict = {}
    fk_waived: set = set()
    prev_stats: str = ""
    for i in range(1, max_iter + 1):
        rejection_reasons: list[str] = []
        if prev_stats:
            for line in prev_stats.splitlines():
                console.print(f"[dim]  {line}[/dim]")
            console.print()
            prev_stats = ""
        console.print(f"[dim]  Phase {phase_label} — iteration {i}/{max_iter}…[/dim]")

        call_start = time.monotonic()
        if output_bucket is not None:
            waited = output_bucket.proactive_wait(now=call_start, expected=last_out_tokens)
            if waited > 0:
                console.print(
                    f"[yellow]  Output token budget near limit — waited {waited:.1f}s "
                    f"(proactive throttle)[/yellow]"
                )

        response = _call_with_retry(backend, tools)
        now = time.monotonic()
        iter_duration = now - call_start

        if tracker is not None:
            tracker.record(now=now)

        iter_usage = backend.extract_usage(response)
        for key, val in iter_usage.items():
            usage[key] += val

        iter_in           = iter_usage.get("input_tokens", 0)
        iter_out          = iter_usage.get("output_tokens", 0)
        iter_cache_create = iter_usage.get("cache_creation_tokens", 0)
        iter_cache_read   = iter_usage.get("cache_read_tokens", 0)

        if output_bucket is not None:
            output_bucket.record(now=now, tokens=iter_out)
            rl = backend.last_output_rate_limit()
            if rl.get("limit"):
                new_limit = rl["limit"]
                if new_limit != output_bucket._limit:
                    console.print(
                        f"[dim]  Output token limit updated from API headers: "
                        f"{output_bucket._limit:,} → {new_limit:,} tokens/min[/dim]"
                    )
                output_bucket.update_limit(new_limit)
        _p = get_model_pricing(MODEL)
        from schematica.pricing import CACHE_WRITE_MULTIPLIER, CACHE_READ_MULTIPLIER
        usage["total_cost"] += (iter_in * _p["input"] + iter_out * _p["output"]) / 1_000_000
        if iter_cache_create:
            usage["total_cost"] += iter_cache_create * _p.get("cache_write", _p["input"] * CACHE_WRITE_MULTIPLIER) / 1_000_000
        if iter_cache_read:
            usage["total_cost"] += iter_cache_read * _p.get("cache_read", _p["input"] * CACHE_READ_MULTIPLIER) / 1_000_000
        prev_stats = _format_iter_stats(
            iter_in, iter_out, MODEL,
            tracker=tracker, now=now,
            iter_duration=iter_duration,
            total_in=usage.get("input_tokens", 0),
            total_out=usage.get("output_tokens", 0),
            total_cost=usage["total_cost"],
            iter_num=i,
            max_iter=max_iter,
            context_window=_context_window(MODEL),
            cache_creation_tokens=iter_cache_create,
            cache_read_tokens=iter_cache_read,
        )

        backend.append_assistant(response)

        # If the model ran out of tokens the tool call JSON is truncated — tell
        # the agent explicitly so it can retry rather than silently missing keys.
        if backend.stop_reason(response) == "max_tokens":
            console.print(
                f"[yellow]  max_tokens hit in phase {phase_label} — output was truncated. "
                "Asking agent to retry.[/yellow]"
            )
            # Compact the truncated message so repeated failed attempts don't balloon the context.
            backend.compress_truncated()
            # The truncated response may contain tool_use blocks with no corresponding
            # tool_result. The API requires every tool_use to be immediately followed by
            # a tool_result, so inject error results for each orphaned tool_use block.
            if not backend.append_orphaned_errors(response):
                backend.append_user(
                    "Your previous response was truncated because it exceeded the output token limit. "
                    "Your finish_catalogue JSON was cut off before all fields were written. "
                    "Resubmit finish_catalogue — write the JSON more concisely: "
                    "shorten agent_notes and descriptions to 1–2 sentences each, "
                    "keep SQL queries on one line, and include every required field "
                    "(tables, measurable_metrics, queryable_facts, time_coverage, "
                    "data_quality_notes, description)."
                )
            continue

        # Collect any tool_use blocks regardless of stop_reason
        tool_use_blocks = backend.tool_calls(response)

        if not tool_use_blocks:
            if i < min_iter:
                # Still in the mandatory exploration window — nudge to keep exploring
                console.print(f"[yellow]  No tool calls in phase {phase_label} iteration {i} — nudging to continue[/yellow]")
                backend.append_user(
                    "You must keep exploring. Call run_query to investigate more tables and "
                    "cross-table JOIN opportunities before calling finish_catalogue. "
                    f"You have used {i}/{max_iter} iterations; minimum required is {min_iter}."
                )
                continue
            if i < max_iter:
                # Past the minimum but haven't submitted yet — nudge to finish
                console.print(f"[yellow]  No tool calls in phase {phase_label} iteration {i} — nudging to submit[/yellow]")
                backend.append_user(
                    "You have explored enough. Call finish_catalogue now with everything "
                    "you have learned. Include all tables, measurable_metrics, queryable_facts, "
                    "time_coverage, data_quality_notes, description, and overview."
                )
                continue
            console.print(f"[yellow]  Agent produced no tool calls in phase {phase_label}[/yellow]")
            return None, last_rejection_reasons, i

        # Process all tool_use blocks and collect results as (tool_id, content) pairs
        pending_results: list[tuple[str, str]] = []
        catalogue_data = None

        # Run all run_query blocks concurrently, then print results in order.
        # executor.map preserves input order so zip is safe.
        run_query_blocks = [b for b in tool_use_blocks if b.name == "run_query"]
        for block, (block_id, result) in zip(
            run_query_blocks, _run_queries_parallel(engine, run_query_blocks)
        ):
            _print_query(
                sql=block.input["sql"],
                reason=block.input.get("reason", ""),
                tables=block.input.get("tables", []),
                columns=block.input.get("columns", []),
                plain_language=block.input.get("plain_language", ""),
                result=result,
                table_columns=table_columns,
            )
            pending_results.append((block_id, result))

        for block in tool_use_blocks:
            if block.name == "run_query":
                continue  # already handled above

            elif block.name == "finish_catalogue":
                # Normalise fields the model may have JSON-encoded as strings.
                block.input = _coerce_json_strings(block.input)
                # Coerce data_quality_notes to a list of strings — models sometimes
                # submit a count, a single string, or a list of non-strings.
                # If it's not a proper list of strings, discard it so the
                # "empty notes" warning fires and the agent adds real notes.
                dqn = block.input.get("data_quality_notes")
                if not isinstance(dqn, list) or not all(isinstance(n, str) for n in dqn):
                    block.input["data_quality_notes"] = []
                required = {"tables", "measurable_metrics", "queryable_facts", "time_coverage", "description", "overview"}
                missing = required - block.input.keys()
                empty_required = [
                    k for k in ("tables", "measurable_metrics")
                    if k in block.input and not block.input[k]
                ]
                rejection_reasons = []
                # tables submitted as a non-list type (e.g. entire catalogue JSON as string)
                raw_tables = block.input.get("tables", [])
                if not isinstance(raw_tables, list):
                    block.input["tables"] = []
                    rejection_reasons.append(
                        f"tables is {type(raw_tables).__name__!r}, not a list. "
                        + _TABLES_NOT_LIST_ERROR_MSG
                    )
                # tables submitted as bare name strings instead of TableSummary objects
                elif raw_tables and isinstance(raw_tables[0], str):
                    block.input["tables"] = []
                    rejection_reasons.append(
                        f"tables[0] is {raw_tables[0]!r} (a string). " + _BARE_TABLES_ERROR_MSG
                    )
                if block.input.get("_compressed"):
                    rejection_reasons.append(
                        "Your previous finish_catalogue was truncated mid-response because the JSON "
                        "was too large for the output token limit. You must resubmit a smaller catalogue. "
                        "Cut it down to the most important metrics: aim for 3–5 per table, ~70 total. "
                        "Drop low-value breakdowns and near-duplicate metrics. "
                        "Keep agent_notes and descriptions to one sentence each. "
                        "Every item must be a full object with all required fields "
                        "(name, description, sql, time_range, granularity, unit, tables_used, "
                        "confidence, agent_notes) — not just a name."
                    )
                if not rejection_reasons and i < min_iter:
                    remaining = min_iter - i
                    console.print(
                        f"[bold yellow]  ⚠ finish_catalogue called too early "
                        f"(iteration {i}/{max_iter}, min {min_iter}) — "
                        f"forcing {remaining} more iteration(s)[/bold yellow]"
                    )
                    rejection_reasons.append(
                        f"Too early — you have used only {i}/{max_iter} iterations "
                        f"(minimum is {min_iter}). "
                        f"You MUST call run_query next — do NOT call finish_catalogue again yet. "
                        f"Look for cross-table JOIN opportunities using the FK relationships "
                        f"visible in the schema. For each FK, run at least one query that groups "
                        f"by a period column to validate whether a time-series metric is achievable."
                    )
                if missing:
                    rejection_reasons.append(
                        f"Keys absent from submission: {sorted(missing)}"
                    )
                if empty_required:
                    rejection_reasons.append(
                        f"Required lists submitted as empty: {sorted(empty_required)} — "
                        "you explored this database and found data; include all metrics and tables you discovered"
                    )
                all_items = list(block.input.get("measurable_metrics", [])) + list(block.input.get("queryable_facts", []))
                tables_used_errors = _tables_used_violations(all_items)
                if tables_used_errors:
                    rejection_reasons.append(
                        "tables_used mismatch — tables_used must list only tables that appear "
                        "in the SQL FROM/JOIN clauses:\n  " + "\n  ".join(tables_used_errors)
                    )
                # FK coverage: every FK relationship must have at least one metric
                # whose SQL JOINs both tables. Pairs that have been rejected
                # _FK_REJECTION_CAP times are waived to prevent infinite loops.
                if fk_pairs:
                    submitted_metrics = [m for m in block.input.get("measurable_metrics", []) if isinstance(m, dict)]
                    effective_fk_pairs = [p for p in fk_pairs if frozenset(p) not in fk_waived]
                    missing_fks = _uncovered_fk_pairs(submitted_metrics, effective_fk_pairs, lookup_tables)
                    if missing_fks:
                        fk_rejection_counts, fk_waived = _update_fk_waived(
                            missing_fks, fk_rejection_counts, fk_waived
                        )
                        pairs_str = ", ".join(f"{a}↔{b}" for a, b in missing_fks)
                        rejection_reasons.append(_FK_REJECTION_MSG.format(pairs_str=pairs_str))
                # Pre-validate metrics and facts against the Pydantic schema so
                # validation errors are returned to the agent rather than crashing.
                if not rejection_reasons:
                    from schematica.catalogue import MeasurableMetric, QueryableFact
                    schema_errors = []
                    for idx, m in enumerate(block.input.get("measurable_metrics", [])):
                        if not isinstance(m, dict):
                            continue
                        try:
                            MeasurableMetric.model_validate(m)
                        except Exception as e:
                            schema_errors.append(f"measurable_metrics[{idx}] ({m.get('name', '?')}): {e}")
                    for idx, f in enumerate(block.input.get("queryable_facts", [])):
                        if not isinstance(f, dict):
                            continue
                        try:
                            QueryableFact.model_validate(f)
                        except Exception as e:
                            schema_errors.append(f"queryable_facts[{idx}] ({f.get('name', '?')}): {e}")
                    if schema_errors:
                        rejection_reasons.append(
                            "Schema validation errors — fix these field values:\n  "
                            + "\n  ".join(schema_errors)
                        )
                _print_finish_catalogue(block.input, accepted=not bool(rejection_reasons))
                if rejection_reasons:
                    reasons_text = "; ".join(rejection_reasons)
                    console.print(f"[yellow]  finish_catalogue rejected — {reasons_text}[/yellow]")
                    # Replace the full catalogue payload in the assistant message with a
                    # compact summary so it doesn't bloat the context on every subsequent
                    # iteration. The tool_result feedback is sufficient for the agent to
                    # know what was missing.
                    _raw_tables  = block.input.get("tables")
                    _raw_metrics = block.input.get("measurable_metrics")
                    _raw_facts   = block.input.get("queryable_facts")
                    _compress_summary = {
                        "_compressed": True,
                        "description": (block.input.get("description") or "")[:120],
                        "tables": [t.get("name", "?") if isinstance(t, dict) else str(t) for t in _raw_tables] if isinstance(_raw_tables, list) else [],
                        "measurable_metrics": [m.get("name", "?") if isinstance(m, dict) else str(m) for m in _raw_metrics] if isinstance(_raw_metrics, list) else [],
                        "queryable_facts": [f.get("name", "?") if isinstance(f, dict) else str(f) for f in _raw_facts] if isinstance(_raw_facts, list) else [],
                        "time_coverage": block.input.get("time_coverage"),
                        "data_quality_notes": len(block.input.get("data_quality_notes") or []),
                    }
                    backend.compress_finish_catalogue(block.id, _compress_summary)
                    pending_results.append((
                        block.id,
                        f"ERROR: {reasons_text}. "
                        "Resubmit finish_catalogue with all required fields present and non-empty "
                        "where data was found: tables, measurable_metrics, time_coverage, "
                        "data_quality_notes, description, overview. "
                        "queryable_facts may be empty if none were found.",
                    ))
                else:
                    catalogue_data = block.input
                    dq_notes = block.input.get("data_quality_notes") or []
                    if not dq_notes:
                        console.print(
                            "[yellow]  ⚠ data_quality_notes is empty — "
                            "accepted, but the schema snapshot likely shows nullable columns "
                            "or partial coverage worth documenting.[/yellow]"
                        )
                        acceptance_msg = (
                            "Catalogue accepted. "
                            "NOTE: data_quality_notes is empty. Real databases almost always "
                            "have nullable columns or partial coverage that metric consumers "
                            "need to know about. Review the schema snapshot and add relevant "
                            "notes in a follow-up if you missed any."
                        )
                    else:
                        acceptance_msg = "Catalogue accepted."
                    pending_results.append((block.id, acceptance_msg))

        # Always append tool_results to keep the message history valid
        backend.append_tool_results(pending_results)
        backend.compress_old_run_queries()
        last_out_tokens = iter_out

        if catalogue_data is not None:
            if prev_stats:
                for line in prev_stats.splitlines():
                    console.print(f"[dim]  {line}[/dim]")
            console.print(f"[green]  finish_catalogue called in phase {phase_label}, iteration {i}.[/green]")
            return catalogue_data, [], i

        # Track last rejection reasons for the caller to use in the next phase prompt
        if rejection_reasons:
            last_rejection_reasons = rejection_reasons

        # If stop_reason was end_turn despite having tool_use blocks, exit phase
        if backend.stop_reason(response) == "end_turn":
            if prev_stats:
                for line in prev_stats.splitlines():
                    console.print(f"[dim]  {line}[/dim]")
            return None, last_rejection_reasons, i

    if prev_stats:
        for line in prev_stats.splitlines():
            console.print(f"[dim]  {line}[/dim]")
    return None, last_rejection_reasons, max_iter


# ── query execution ────────────────────────────────────────────────────────────

def _snap_to_period(date_str: str, granularity: str) -> str:
    """Truncate a date string to the natural boundary of a granularity."""
    if not date_str:
        return date_str
    if len(date_str) == 4:
        # Bare year e.g. "2009" or "1871" — always expand to YYYY-01-01
        return date_str + "-01-01"
    if len(date_str) < 7:
        return date_str
    if granularity in ("monthly", "quarterly"):
        return date_str[:7] + "-01"
    if granularity == "annual":
        return date_str[:4] + "-01-01"
    return date_str


def _execute_query(engine, sql: str, reason: str) -> str:
    # Strip leading SQL line comments (-- ...) before the SELECT check so the
    # agent can include explanatory comments without triggering the guard.
    sql_stripped = "\n".join(
        line for line in sql.splitlines() if not line.strip().startswith("--")
    ).strip()
    if not sql_stripped.upper().startswith("SELECT"):
        return "ERROR: Only SELECT queries are allowed."
    sql = sql_stripped
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchmany(MAX_ROWS)
            cols = list(result.keys())

            if not rows:
                return "Query returned 0 rows."

            lines = [" | ".join(cols), "-" * max(len(" | ".join(cols)), 10)]
            for row in rows:
                lines.append(" | ".join(str(v) for v in row))

            output = "\n".join(lines)
            if len(rows) == MAX_ROWS:
                output += f"\n(truncated to {MAX_ROWS} rows)"
            if len(output) > MAX_CHARS:
                output = output[:MAX_CHARS] + "\n... (truncated)"
            return output

    except Exception as exc:
        return f"ERROR: {exc}"


def _run_queries_parallel(engine, run_query_blocks: list) -> list[tuple[str, str]]:
    """Execute a list of run_query tool-use blocks concurrently.

    Returns a list of (tool_id, result_string) pairs in the same order as the
    input blocks.  Uses a thread pool capped at the number of blocks; SQLAlchemy
    engines are thread-safe and manage their own connection pool.
    """
    if not run_query_blocks:
        return []

    def _run(block) -> tuple[str, str]:
        result = _execute_query(engine, block.input["sql"], block.input.get("reason", ""))
        return (block.id, result)

    with ThreadPoolExecutor(max_workers=len(run_query_blocks)) as executor:
        return list(executor.map(_run, run_query_blocks))


# ── output construction ────────────────────────────────────────────────────────

def _build_catalogue(data: dict, snapshot: dict) -> DataCatalogue:
    try:
        return DataCatalogue(
            analysed_at=datetime.now().isoformat(timespec="seconds"),
            model=MODEL,
            connection=snapshot["connection_string"],
            dialect=snapshot["dialect"],
            description=data.get("description") or "",
            overview=data.get("overview") or "",
            tables=data["tables"],
            measurable_metrics=data["measurable_metrics"],
            queryable_facts=data.get("queryable_facts") or [],
            time_coverage=data["time_coverage"],
            data_quality_notes=data.get("data_quality_notes") or [],
            key_terms=data.get("key_terms") or [],
            table_relationships=data.get("table_relationships") or [],
        )
    except Exception as exc:
        console.print(f"[red]Catalogue validation error: {exc}[/red]")
        console.print(f"[dim]Keys received: {list(data.keys())}[/dim]")
        raise


def _decode_sql_error(exc: Exception) -> str:
    """Extract a concise, readable root cause from a SQLAlchemy exception."""
    msg = str(exc)
    # SQLAlchemy wraps the DB error — strip the ORM preamble to surface the actual cause
    for marker in ("(sqlite3.OperationalError)", "(psycopg2.", "(pymysql.err.", "sqlalchemy.exc."):
        idx = msg.find(marker)
        if idx != -1:
            msg = msg[idx:]
            break
    return msg.split("\n")[0][:200]


def _probe_connection(engine, connection_string: str) -> None:
    """Verify the database is reachable before spending tokens on introspection.

    SQLite is exempt — its file-existence check runs earlier in run().
    For all other dialects, attempt a trivial connection and raise SystemExit
    with a clear message if it fails.
    """
    if connection_string.startswith("sqlite"):
        return
    try:
        with engine.connect():
            pass
    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] Cannot connect to database: {exc}\n"
            f"  Connection string: {connection_string}\n"
            "  Check that the database is running, the credentials are correct, "
            "and the host is reachable."
        )
        raise SystemExit(1)


def _select_phase3_result(
    refined: DataCatalogue,
    patched: DataCatalogue,
) -> tuple[DataCatalogue, bool]:
    """
    Choose between the refined catalogue and the fallback (directly-patched) catalogue.

    Returns (chosen_catalogue, did_fallback).
    Falls back to patched when refined is completely empty — this happens when
    _drop_broken_sql removes all metrics because the refinement agent hallucinated
    column names or submitted otherwise unrunnable SQL.
    """
    if not refined.measurable_metrics and not refined.queryable_facts:
        return patched, True
    return refined, False


def _drop_broken_sql(catalogue: DataCatalogue, engine) -> DataCatalogue:
    """
    Run each metric and queryable fact SQL against the engine with LIMIT 1.
    Any that fail are removed from the catalogue with a warning.
    This is a syntax/validity check only — full row-level evaluation happens in Phase 3.
    """
    good_metrics = []
    for m in catalogue.measurable_metrics:
        try:
            with engine.connect() as conn:
                conn.execute(text(f"SELECT * FROM ({m.sql}) LIMIT 1"))
            good_metrics.append(m)
        except Exception as exc:
            console.print(
                f"[yellow]  ⚠ Dropping metric [bold]{m.name}[/bold] — SQL syntax check failed "
                f"(metrics must return exactly 2 columns: date + value; this one cannot run).\n"
                f"    Cause: {_decode_sql_error(exc)}[/yellow]"
            )

    good_facts = []
    for f in catalogue.queryable_facts:
        try:
            with engine.connect() as conn:
                conn.execute(text(f"SELECT * FROM ({f.sql}) LIMIT 1"))
            good_facts.append(f)
        except Exception as exc:
            console.print(
                f"[yellow]  ⚠ Dropping fact [bold]{f.name}[/bold] — SQL syntax check failed "
                f"(queryable facts are non-time-series snapshots, e.g. lookup tables or current-state queries; "
                f"this one cannot run and will be excluded from the catalogue).\n"
                f"    Cause: {_decode_sql_error(exc)}[/yellow]"
            )

    return catalogue.model_copy(
        update={"measurable_metrics": good_metrics, "queryable_facts": good_facts}
    )


def _render_overview_md(catalogue: "DataCatalogue") -> str:
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

    def _metric_group(m: "MeasurableMetric") -> str:
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


def _write_output(catalogue: DataCatalogue, out_path: str) -> None:
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


# ── phase 3 — refinement ───────────────────────────────────────────────────────

_DIRECTLY_PATCHED = {"date_mismatch", "period_boundary"}


def _filter_agent_issues(
    warn_metrics: list[tuple],
    warn_facts:   list[tuple],
) -> list[tuple]:
    """Return only the (item, result) pairs that need agent investigation.

    Excluded:
      - Metrics whose only issues are date_mismatch / period_boundary (patched directly).
      - Any result where the SQL ran fine but the eval framework crashed
        (sql_ok=True, error starts with "eval error:") — the agent cannot fix
        an environment problem like a missing numpy module.
    """
    return (
        [
            (m, r) for m, r in warn_metrics
            if set((r.error or "").split(", ")) - _DIRECTLY_PATCHED
            and not _is_evaluator_crash(r)
        ] +
        [(f, r) for f, r in warn_facts if not _is_evaluator_crash(r)]
    )

def _run_phase3_safe(
    phase3_fn,
    catalogue: "DataCatalogue",
    schema_text: str,
    engine,
    usage: dict,
    table_columns: dict,
    **kwargs,
) -> tuple:
    """Call phase3_fn; on any exception fall back to the Phase 1 catalogue.

    Ensures _write_output is always reached in run() even if Phase 3 crashes
    (network error, FileNotFoundError, evaluator crash, etc.).
    Returns (catalogue, metric_results, fact_results, uncovered_tables).
    """
    try:
        return phase3_fn(catalogue, schema_text, engine, usage, table_columns, **kwargs)
    except Exception as exc:
        console.print(
            f"\n[yellow]  Phase 3 failed: {exc}[/yellow]\n"
            "  [dim]Writing Phase 1 catalogue as fallback.[/dim]"
        )
        return catalogue, [], [], []


def _run_phase3(
    catalogue: DataCatalogue,
    schema_text: str,
    engine,
    usage: dict,
    table_columns: dict,
    tracker: "_RequestTracker | None" = None,
) -> tuple[DataCatalogue, list, list, list[str]]:
    """
    Evaluate the catalogue against the live database, patch trivial issues
    directly, then run a refinement agent loop for anything that needs
    investigation. Returns (catalogue, metric_results, fact_results, uncovered_tables).
    """
    console.print(Panel(
        "[bold cyan]Phase 3 — Refinement[/bold cyan]\n"
        "[dim]Evaluating catalogue against live database…[/dim]",
        border_style="cyan",
        padding=(0, 1),
    ))

    # ── run eval ───────────────────────────────────────────────────────────────
    # Each SQL is run against the full database (no row limit) so row counts,
    # null rates, and date ranges are exact — not sampled.
    console.print(
        f"[dim]  Running {len(catalogue.measurable_metrics)} metric(s) and "
        f"{len(catalogue.queryable_facts)} fact(s) against the full database…[/dim]"
    )
    metric_results = [evaluate_metric(engine, m.model_dump()) for m in catalogue.measurable_metrics]
    fact_results   = [evaluate_fact(engine, f.model_dump())   for f in catalogue.queryable_facts]

    warn_metrics = [
        (catalogue.measurable_metrics[i], metric_results[i])
        for i, r in enumerate(metric_results) if r.status in ("WARN", "FAIL")
    ]
    warn_facts = [
        (catalogue.queryable_facts[i], fact_results[i])
        for i, r in enumerate(fact_results) if r.status in ("WARN", "FAIL")
    ]

    # ── check table coverage ───────────────────────────────────────────────────
    from sqlalchemy import inspect as sqla_inspect
    db_tables  = set(sqla_inspect(engine).get_table_names())
    mentioned  = {t for m in catalogue.measurable_metrics for t in m.tables_used}
    mentioned |= {t for f in catalogue.queryable_facts    for t in f.tables_used}
    mentioned |= {t.name for t in catalogue.tables}
    uncovered  = sorted(db_tables - mentioned)
    if uncovered:
        console.print(
            f"  [yellow]⚠ Tables with no metrics or facts:[/yellow] {', '.join(uncovered)}"
        )

    if not warn_metrics and not warn_facts:
        console.print("[green]  No metric/fact issues found — skipping refinement.[/green]\n")
        return catalogue, metric_results, fact_results, uncovered

    # ── display issues ─────────────────────────────────────────────────────────
    console.print(f"  [bold]Issues found ({len(warn_metrics) + len(warn_facts)}):[/bold]")
    for m, r in warn_metrics:
        console.print(f"    [yellow]⚠ {r.status}[/yellow]  {m.name}  —  {r.error}")
    for f, r in warn_facts:
        console.print(f"    [yellow]⚠ {r.status}[/yellow]  (fact) {f.name}  —  {r.error}")

    # Print a legend for only the warn codes that actually appear in this run
    seen_codes: set[str] = set()
    for _, r in warn_metrics + warn_facts:
        for code in (r.error or "").split(", "):
            code = code.strip()
            if code in _PHASE3_WARN_LEGEND:
                seen_codes.add(code)
    if seen_codes:
        console.print("  [dim]  Legend:[/dim]")
        for code in sorted(seen_codes):
            console.print(f"    [dim]{code}: {_PHASE3_WARN_LEGEND[code]}[/dim]")
    console.print()

    # ── patch date_mismatch and period_boundary directly — no LLM needed ────────
    from schematica.catalogue import TimeRange
    patched_metrics = {m.name: m for m in catalogue.measurable_metrics}
    date_patched: list[str] = []

    for m, r in warn_metrics:
        codes = set((r.error or "").split(", "))

        if "date_mismatch" in codes and r.actual_start and r.actual_end:
            new_start = r.actual_start
            new_end   = r.actual_end
        else:
            new_start = m.time_range.start
            new_end   = m.time_range.end

        if "period_boundary" in codes:
            new_start = _snap_to_period(new_start, m.granularity)
            new_end   = _snap_to_period(new_end,   m.granularity)

        if new_start != m.time_range.start or new_end != m.time_range.end:
            patched_metrics[m.name] = m.model_copy(update={
                "time_range": TimeRange(start=new_start, end=new_end)
            })
            date_patched.append(
                f"    [green]✓[/green] {m.name}: "
                f"[dim]{m.time_range.start[:7]} → {m.time_range.end[:7]}[/dim] → "
                f"[green]{new_start[:7]} → {new_end[:7]}[/green]"
                + (f" [dim](period_boundary)[/dim]" if "period_boundary" in codes else "")
            )

    if date_patched:
        console.print(
            "  [bold]Patching date mismatches directly:[/bold]\n"
            "  [dim]Old values = declared by agent (from schema snapshot min/max).\n"
            "  New values = measured by Phase 3 eval running the full SQL against the live DB.[/dim]"
        )
        for line in date_patched:
            console.print(line)
        console.print()

    # ── collect issues needing agent investigation ─────────────────────────────
    agent_issues = _filter_agent_issues(warn_metrics, warn_facts)

    if not agent_issues:
        console.print("  [green]All issues patched directly — no agent call needed.[/green]\n")
        final = catalogue.model_copy(update={"measurable_metrics": list(patched_metrics.values())})
        final_mr = [evaluate_metric(engine, m.model_dump()) for m in final.measurable_metrics]
        final_fr = [evaluate_fact(engine, f.model_dump())   for f in final.queryable_facts]
        return final, final_mr, final_fr, uncovered

    # ── build refinement prompt ────────────────────────────────────────────────
    issues_text = ""
    for item, r in agent_issues:
        issues_text += (
            f"\n{'METRIC' if hasattr(item, 'time_range') else 'FACT'}: {item.name}\n"
            f"  Issue: {r.error}\n"
            f"  SQL: {item.sql}\n"
        )

    current_catalogue_json = json.dumps(
        catalogue.model_copy(update={
            "measurable_metrics": list(patched_metrics.values())
        }).model_dump(),
        indent=2,
    )

    refinement_prompt = (
        f"Here is the schema snapshot:\n\n```\n{schema_text}\n```\n\n"
        f"Here is the current catalogue:\n\n```json\n{current_catalogue_json}\n```\n\n"
        f"The following issues were found by the evaluator:\n{issues_text}\n"
        f"Investigate each issue using run_query, fix what you can, "
        f"then submit the complete corrected catalogue via finish_catalogue."
    )

    console.print(Panel(
        f"[bold]Feeding back to agent — {len(agent_issues)} issue(s) to investigate[/bold]\n"
        "[dim]These issues require data investigation (wrong filter, missing rows, extra columns) "
        "and cannot be fixed by date arithmetic alone. The agent will use run_query to inspect "
        "the data and resubmit a corrected catalogue via finish_catalogue.\n"
        "Source: Phase 3 eval ran each SQL against the full DB and flagged these results.[/dim]\n\n"
        + "\n".join(
            f"  [yellow]⚠[/yellow]  {'(fact) ' if not hasattr(item, 'time_range') else ''}"
            f"{item.name}  —  {r.error}"
            for item, r in agent_issues
        ),
        border_style="yellow",
        padding=(0, 1),
    ))

    # ── run refinement loop ────────────────────────────────────────────────────
    # Mark the refinement prompt for caching on the Anthropic path — it contains
    # the full schema + catalogue JSON and is repeated across all refinement iterations.
    backend = _make_backend(refinement_prompt, REFINEMENT_SYSTEM_PROMPT)

    refined_data, _, _ = _run_phase(
        backend, engine,
        tools=[_RUN_QUERY_TOOL, _FINISH_CATALOGUE_TOOL],
        max_iter=_REFINEMENT_BUDGET,
        phase_label="3 (refinement)",
        usage=usage,
        table_columns=table_columns,
        tracker=tracker,
    )

    if refined_data is None:
        console.print(
            "[yellow]  Refinement agent did not produce a valid catalogue — "
            "keeping directly-patched catalogue.[/yellow]"
        )
        final = catalogue.model_copy(update={"measurable_metrics": list(patched_metrics.values())})
        final_mr = [evaluate_metric(engine, m.model_dump()) for m in final.measurable_metrics]
        final_fr = [evaluate_fact(engine, f.model_dump())   for f in final.queryable_facts]
        return final, final_mr, final_fr, uncovered

    refined_catalogue = _build_catalogue(refined_data, {
        "connection_string": catalogue.connection,
        "dialect":           catalogue.dialect,
        "tables":            [],
    })
    refined_catalogue = _drop_broken_sql(refined_catalogue, engine)

    patched_catalogue = catalogue.model_copy(update={"measurable_metrics": list(patched_metrics.values())})
    refined_catalogue, did_fallback = _select_phase3_result(refined_catalogue, patched_catalogue)

    if did_fallback:
        console.print(
            "[yellow]  ⚠ Refinement produced an empty catalogue after SQL validation "
            "(all metrics had broken SQL) — falling back to the directly-patched catalogue.[/yellow]"
        )
        final_mr = [evaluate_metric(engine, m.model_dump()) for m in refined_catalogue.measurable_metrics]
        final_fr = [evaluate_fact(engine, f.model_dump())   for f in refined_catalogue.queryable_facts]
        return refined_catalogue, final_mr, final_fr, uncovered

    n_orig    = len(catalogue.measurable_metrics)
    n_refined = len(refined_catalogue.measurable_metrics)
    if n_orig > 0 and n_refined < n_orig * 0.5:
        console.print(
            f"[yellow]  ⚠ Refinement dropped {n_orig - n_refined} of {n_orig} metrics "
            f"({n_refined} remain). Consider re-running with a larger refinement budget.[/yellow]"
        )

    # ── phase 3 summary ───────────────────────────────────────────────────────
    orig_names    = {m.name for m in catalogue.measurable_metrics}
    refined_names = {m.name for m in refined_catalogue.measurable_metrics}
    removed = orig_names - refined_names
    added   = refined_names - orig_names

    lines = [
        f"  Date mismatches patched directly:  {len(date_patched)}",
        f"  Issues sent to agent:              {len(agent_issues)}",
    ]
    if removed:
        lines.append(f"  Metrics removed:  {', '.join(sorted(removed))}")
    if added:
        lines.append(f"  Metrics added:    {', '.join(sorted(added))}")
    lines.append(
        f"  Final catalogue:  "
        f"{len(refined_catalogue.measurable_metrics)} metrics, "
        f"{len(refined_catalogue.queryable_facts)} facts"
    )

    console.print(Panel(
        "[bold green]Phase 3 complete[/bold green]\n" + "\n".join(lines),
        border_style="green",
        padding=(0, 1),
    ))

    final_mr = [evaluate_metric(engine, m.model_dump()) for m in refined_catalogue.measurable_metrics]
    final_fr = [evaluate_fact(engine, f.model_dump())   for f in refined_catalogue.queryable_facts]
    return refined_catalogue, final_mr, final_fr, uncovered


# ── Anthropic client ───────────────────────────────────────────────────────────

def _get_anthropic_client():
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set. Add it to .env or the environment.")
    return anthropic.Anthropic(api_key=api_key)


# ── rich output helpers ────────────────────────────────────────────────────────

def _print_header(connection_string: str, out_path: str) -> None:
    model_line = MODEL
    if MODEL.startswith("anthropic/"):
        model_line += "  [dim](cache: on)[/dim]" if _CACHE else "  [dim](cache: off — set SC_CACHE=true to enable)[/dim]"
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


def _print_schema_detail(schema_text: str) -> None:
    console.print(Panel(
        f"[dim]{schema_text}[/dim]",
        title="Schema sent to LLM",
        border_style="dim",
        padding=(0, 1),
    ))


def _print_summary(
    catalogue: DataCatalogue,
    usage: dict,
    elapsed_secs: float,
    metric_results: list,
    fact_results: list,
    uncovered_tables: list[str],
) -> None:
    inp           = usage["input_tokens"]
    out           = usage["output_tokens"]
    cache_created = usage.get("cache_creation_tokens", 0)
    cache_read    = usage.get("cache_read_tokens", 0)
    mins, secs = divmod(int(elapsed_secs), 60)
    elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    cost_str = format_cost(MODEL, inp, out, cache_created, cache_read)

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
        f"  Model:                {MODEL}"
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


# ── CLI ────────────────────────────────────────────────────────────────────────

def _model_folder_name() -> str:
    """
    Return a filesystem-safe folder name derived from MODEL.

    'gemini/gemini-2.5-flash'              →  'gemini-2.5-flash'
    'anthropic/claude-haiku-4-5-20251001'  →  'claude-haiku-4-5-20251001'
    """
    # Strip provider prefix (e.g. "gemini/", "anthropic/")
    name = MODEL.split("/", 1)[-1] if "/" in MODEL else MODEL
    # Replace any remaining characters that are not safe in folder names
    return re.sub(r"[^\w.\-]", "_", name)


def _next_catalogue_index(out_dir: Path, db_name: str) -> int:
    """Return the next available 1-based index for <db_name>_catalogue_<n>.json."""
    if not out_dir.exists():
        return 1
    existing = list(out_dir.glob(f"{db_name}_catalogue_*.json"))
    indices = []
    for p in existing:
        # Extract the numeric suffix between the last '_' and '.json'
        stem = p.stem  # e.g. solar_wind_catalogue_3
        suffix = stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            indices.append(int(suffix))
    return max(indices, default=0) + 1


def _derive_catalogue_path(connection_string: str) -> str:
    """
    Derive the catalogue output path from a connection string.

    Catalogues are stored in data/<model>/<db_name>_catalogue_<n>.json,
    where <n> auto-increments so repeated runs never overwrite each other.

    sqlite:///data/solar_wind.db  →  data/gemini-2.5-flash/solar_wind_catalogue_1.json
    postgresql://.../mydb         →  data/gemini-2.5-flash/mydb_catalogue_1.json
    """
    from urllib.parse import urlparse

    if connection_string.startswith("sqlite:///"):
        db_file = Path(connection_string[len("sqlite:///"):])
        db_name = db_file.stem
        data_dir = db_file.parent
    else:
        parsed = urlparse(connection_string)
        db_name = parsed.path.lstrip("/")
        data_dir = Path("data")

    out_dir = data_dir / _model_folder_name()
    idx = _next_catalogue_index(out_dir, db_name)
    return str(out_dir / f"{db_name}_catalogue_{idx}.json")


_KNOWN_SCHEMES = ("sqlite:///", "postgresql://", "mysql://", "mssql://", "oracle://")


def _to_connection_string(db: str) -> str:
    """Convert a file path to a SQLite connection string; pass through existing connection strings."""
    if any(db.startswith(s) for s in _KNOWN_SCHEMES):
        return db
    return f"sqlite:///{db}"


def main() -> None:
    """
    Console script entry point.

    Usage:
      schematica --db path/to/mydb.db
      schematica --db sqlite:///path/to/mydb.db
      schematica --db postgresql://user:pass@host:5432/mydb
      schematica --db path/to/mydb.db --out path/to/custom.json
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Schematica — analyse a database and produce a data catalogue.",
    )
    parser.add_argument("--db", required=True, metavar="DB",
                        help="Database file path (e.g. ./data/mydb.db) or SQLAlchemy connection string")
    parser.add_argument("--out", default=None, metavar="OUTPUT_JSON",
                        help="Path to write the catalogue JSON (default: <db_stem>_catalogue.json)")
    parser.add_argument("--skip-ro-check", action="store_true",
                        help="Skip the read-only user confirmation prompt (for CI / automated use)")
    parser.add_argument("--model", default=None, metavar="MODEL",
                        help="Override SC_MODEL from .env (e.g. gpt-4o, gemini/gemini-2.5-flash)")
    parser.add_argument("--cache", action="store_true", default=False,
                        help="Enable prompt caching (anthropic/ models only). Overrides SC_CACHE=false.")
    args = parser.parse_args()

    if args.model or args.cache:
        _apply_model_override(args.model or MODEL, cache_override=True if args.cache else None)

    connection_string = _to_connection_string(args.db)
    prompt_readonly_confirmation(connection_string, skip=args.skip_ro_check)
    out_path = args.out or _derive_catalogue_path(connection_string)
    print(f"Output → {out_path}", file=sys.stderr)

    try:
        run(connection_string=connection_string, out_path=out_path)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
