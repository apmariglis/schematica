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
from types import SimpleNamespace
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from sqlalchemy import text

from schematica.db import make_readonly_engine, prompt_readonly_confirmation

from schematica.catalogue import DataCatalogue
from schematica.eval import evaluate_metric, evaluate_fact
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
_CACHE = _require_env("SC_CACHE").lower() == "true"
if _CACHE and not MODEL.startswith("anthropic/"):
    raise RuntimeError(
        f"SC_CACHE=true requires a model with the 'anthropic/' prefix, got: {MODEL!r}. "
        "Either set SC_CACHE=false or change SC_MODEL to e.g. anthropic/claude-haiku-4-5-20251001"
    )
# The Anthropic SDK expects the bare model name without the provider prefix.
_ANTHROPIC_MODEL = MODEL[len("anthropic/"):] if MODEL.startswith("anthropic/") else MODEL

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
    """
    # Each alternative captures the table name into a different group;
    # filter out empty strings and normalise to lowercase.
    # next() uses None as default so subquery expressions (e.g. FROM (...) AS sub)
    # that produce all-empty groups don't raise StopIteration / RuntimeError.
    pattern = r'\b(?:FROM|JOIN)\s+(?:"([^"]+)"|`([^`]+)`|\[([^\]]+)\]|(\w+))'
    matches = re.findall(pattern, sql, re.IGNORECASE)
    return {
        name.lower()
        for groups in matches
        if (name := next((g for g in groups if g), None)) is not None
    }


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


def _phase1_budget(n_tables: int) -> int:
    """Exploration iteration budget, scales with database size."""
    return min(_BUDGET_BASE + n_tables * _BUDGET_MULTIPLIER, _BUDGET_CAP)


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
- Prefer monthly aggregations for event-level data.
- Only include SQL that you have validated by running it in Phase 1.
- Be honest about confidence: high = unambiguous columns, medium = inferred \
  from naming/samples, low = significant uncertainty.
- Do not hallucinate column names — only use columns that appear verbatim in the \
  schema snapshot. If a query returns a column error, look up the exact column name \
  in the schema snapshot and fix the SQL before retrying. Never guess.
- Cover all tables in the tables summary, even those with no measurable metrics.
- Include every distinct measurable metric — do not artificially limit the count. \
  A complex database should have more metrics than a simple one.
- time_range start and end must align to the metric's granularity boundary: \
  for monthly metrics both dates must be the first day of a month; for annual \
  metrics the first day of a year. Use the snapshot column min/max and truncate \
  to the nearest period boundary — never use a mid-period date as a range endpoint.
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
                "items": {
                    "type": "object",
                    "properties": {
                        "name":        {"type": "string"},
                        "row_count":   {"type": "integer"},
                        "description": {"type": "string"},
                        "key_columns": {"type": "array", "items": {"type": "string"}},
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
                            "enum": ["daily", "weekly", "monthly", "quarterly", "annual", "event"],
                        },
                        "unit":        {"type": "string"},
                        "tables_used": {"type": "array", "items": {"type": "string"}},
                        "confidence":  {"type": "string", "enum": ["high", "medium", "low"]},
                        "agent_notes": {"type": "string"},
                    },
                    "required": [
                        "name", "description", "sql", "time_range",
                        "granularity", "unit", "tables_used", "confidence", "agent_notes",
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
            "description": {
                "type": "string",
                "description": (
                    "One or two sentences describing what domain or business this database covers "
                    "and what kinds of questions it can answer."
                ),
            },
        },
        "required": [
            "tables", "measurable_metrics", "queryable_facts",
            "time_coverage", "data_quality_notes",
            "description",
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
    _print_header(connection_string, out_path)

    # For SQLite, verify the file exists before spending any tokens.
    if connection_string.startswith("sqlite:///"):
        db_file = connection_string[len("sqlite:///"):]
        if not os.path.exists(db_file):
            console.print(f"[bold red]Error:[/bold red] SQLite database not found: {db_file}")
            raise SystemExit(1)

    console.print("[dim]Introspecting schema…[/dim]")
    snapshot = introspect(connection_string)
    schema_text = render_as_text(snapshot)
    _print_schema_summary(snapshot)
    _print_schema_detail(schema_text)

    table_columns: dict[str, list[str]] = {
        t["name"]: [c["name"] for c in t["columns"]]
        for t in snapshot["tables"]
    }

    engine = make_readonly_engine(connection_string)
    n_tables = len(snapshot["tables"])
    budget = _phase1_budget(n_tables)
    min_iter = max(_MIN_ITER_FLOOR, budget // _MIN_ITER_DIVISOR)
    console.print(f"[dim]Exploration budget: {budget} iterations for {n_tables} tables  (min {min_iter} before finish)[/dim]\n")

    usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_tokens": 0, "cache_read_tokens": 0}
    started_at = time.monotonic()
    try:
        catalogue_data = _agent_loop(schema_text, engine, budget, min_iter, usage, table_columns)
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
    catalogue, final_metric_results, final_fact_results, uncovered_tables = _run_phase3(
        catalogue, schema_text, engine, usage, table_columns
    )
    _write_output(catalogue, out_path)
    _print_summary(catalogue, usage, elapsed_secs, final_metric_results, final_fact_results, uncovered_tables)

    return catalogue


# ── LLM backends ───────────────────────────────────────────────────────────────
#
# Each backend wraps a provider's API and exposes a uniform interface to
# _run_phase. The Anthropic backend uses the native SDK with prompt caching;
# the LiteLLM backend uses the OpenAI-compatible interface (no caching).


class _AnthropicBackend:
    """Anthropic-native backend. Maintains messages in Anthropic format."""

    def __init__(self, client, model: str, system_prompt: str, messages: list):
        self._client = client
        self._model = model
        self._system = system_prompt
        self.messages = messages  # shared, mutated in-place

    def call(self, tools: list, max_tokens: int):
        return self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": self._system, "cache_control": {"type": "ephemeral"}}],
            messages=self.messages,
            tools=tools,
        )

    def extract_usage(self, response) -> dict:
        u = response.usage
        return {
            "input_tokens":          getattr(u, "input_tokens",                0),
            "output_tokens":         getattr(u, "output_tokens",               0),
            "cache_creation_tokens": getattr(u, "cache_creation_input_tokens", 0),
            "cache_read_tokens":     getattr(u, "cache_read_input_tokens",     0),
        }

    def stop_reason(self, response) -> str:
        return response.stop_reason  # "tool_use" | "end_turn" | "max_tokens"

    def tool_calls(self, response) -> list:
        return [b for b in response.content if getattr(b, "type", None) == "tool_use"]

    def append_assistant(self, response) -> None:
        self.messages.append({"role": "assistant", "content": response.content})

    def append_tool_results(self, results: list[tuple[str, str]]) -> None:
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": content}
            for tid, content in results
        ]})

    def append_orphaned_errors(self, response) -> bool:
        """Inject tool_results for any orphaned tool_use blocks. Returns True if any found."""
        orphaned = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        if not orphaned:
            return False
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": b.id,
             "content": "Response was truncated before this tool call completed. Please retry."}
            for b in orphaned
        ]})
        return True

    def append_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def compress_finish_catalogue(self, block_id: str, summary: dict) -> None:
        last = self.messages[-1]
        if last.get("role") != "assistant":
            return
        for block in last.get("content", []):
            if getattr(block, "type", None) == "tool_use" and block.id == block_id:
                block.input = summary
                return

    def compress_truncated(self) -> None:
        """Replace tool_use inputs in the last (truncated) assistant message with a tiny placeholder."""
        if not self.messages or self.messages[-1].get("role") != "assistant":
            return
        for block in self.messages[-1].get("content", []):
            if getattr(block, "type", None) == "tool_use" and isinstance(block.input, dict):
                block.input = {"_truncated": True, "keys": list(block.input.keys())}


class _LiteLLMBackend:
    """LiteLLM backend. Maintains messages in OpenAI format."""

    def __init__(self, model: str, system_prompt: str, messages: list):
        self._model = model
        self._system = system_prompt
        self.messages = messages  # already in OpenAI format

    def call(self, tools: list, max_tokens: int):
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm is required for non-Anthropic providers. "
                "Install it with: uv sync --extra litellm"
            )
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name":        t["name"],
                    "description": t.get("description", ""),
                    "parameters":  t["input_schema"],
                },
            }
            for t in tools
        ]
        response = litellm.completion(
            model=self._model,
            messages=[{"role": "system", "content": self._system}] + self.messages,
            tools=openai_tools,
            max_tokens=max_tokens,
        )
        if not response.choices:
            raise ValueError(
                f"LiteLLM returned empty choices (model={self._model}). "
                "This usually means the response was filtered or the provider returned an error."
            )
        return response

    def extract_usage(self, response) -> dict:
        u = response.usage
        return {
            "input_tokens":          getattr(u, "prompt_tokens",     0),
            "output_tokens":         getattr(u, "completion_tokens", 0),
            "cache_creation_tokens": 0,
            "cache_read_tokens":     0,
        }

    def _choice(self, response):
        if not response.choices:
            raise ValueError(f"LiteLLM returned empty choices (model={self._model}). "
                             "This usually means the response was filtered or the provider returned an error.")
        return response.choices[0]

    def stop_reason(self, response) -> str:
        reason = self._choice(response).finish_reason
        if reason == "length":      return "max_tokens"
        if reason == "tool_calls":  return "tool_use"
        return "end_turn"

    def tool_calls(self, response) -> list:
        tcs = self._choice(response).message.tool_calls or []
        result = []
        for tc in tcs:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            result.append(SimpleNamespace(
                type="tool_use",
                id=tc.id,
                name=tc.function.name,
                input=args,
            ))
        return result

    def append_assistant(self, response) -> None:
        msg = self._choice(response).message
        d: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        self.messages.append(d)

    def append_tool_results(self, results: list[tuple[str, str]]) -> None:
        for tid, content in results:
            self.messages.append({"role": "tool", "tool_call_id": tid, "content": content})

    def append_orphaned_errors(self, response) -> bool:
        tcs = self._choice(response).message.tool_calls or []
        if not tcs:
            return False
        for tc in tcs:
            self.messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": "Response was truncated before this tool call completed. Please retry.",
            })
        return True

    def append_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def compress_finish_catalogue(self, block_id: str, summary: dict) -> None:
        last = self.messages[-1]
        if last.get("role") != "assistant":
            return
        for tc in last.get("tool_calls", []):
            if isinstance(tc, dict) and tc.get("id") == block_id:
                tc["function"]["arguments"] = json.dumps(summary)
                return

    def compress_truncated(self) -> None:
        """Replace tool_call arguments in the last (truncated) assistant message with a tiny placeholder."""
        if not self.messages or self.messages[-1].get("role") != "assistant":
            return
        for tc in self.messages[-1].get("tool_calls", []):
            if not isinstance(tc, dict):
                continue
            args_str = tc.get("function", {}).get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                keys = list(args.keys()) if isinstance(args, dict) else []
            except (json.JSONDecodeError, TypeError):
                keys = []
            tc["function"]["arguments"] = json.dumps({"_truncated": True, "keys": keys})
        self.messages[-1]["content"] = ""


def _call_with_retry(backend, tools: list, max_tokens: int = _MAX_OUTPUT_TOKENS, max_attempts: int = 7):
    """Call backend.call() with exponential backoff on rate-limit errors."""
    delay = 30
    for attempt in range(1, max_attempts + 1):
        try:
            return backend.call(tools, max_tokens)
        except Exception as exc:
            msg = str(exc).lower()
            is_rate_limit = "rate limit" in msg or "ratelimit" in msg or "429" in msg or "rate_limited" in msg
            is_transient = "empty choices" in msg or "overloaded" in msg or "503" in msg or "502" in msg
            if not (is_rate_limit or is_transient) or attempt == max_attempts:
                raise
            console.print(
                f"[yellow]  Rate limit hit — waiting {delay}s before retry "
                f"(attempt {attempt}/{max_attempts - 1})[/yellow]"
            )
            time.sleep(delay)
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

def _agent_loop(schema_text: str, engine, phase1_budget: int, phase1_min_iter: int, usage: dict, table_columns: dict) -> dict:
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
    catalogue_data, last_rejection_reasons = _run_phase(
        backend, engine,
        tools=[_RUN_QUERY_TOOL, _FINISH_CATALOGUE_TOOL],
        max_iter=phase1_budget,
        min_iter=phase1_min_iter,
        phase_label="1 (exploration)",
        usage=usage,
        table_columns=table_columns,
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

    catalogue_data, _ = _run_phase(
        backend, engine,
        tools=[_FINISH_CATALOGUE_TOOL],
        max_iter=5,
        phase_label="2 (documentation)",
        usage=usage,
        table_columns=table_columns,
    )
    if catalogue_data is not None:
        return catalogue_data

    raise RuntimeError("Agent did not produce a catalogue after both phases.")


def _run_phase(backend, engine, tools: list, max_iter: int, phase_label: str, usage: dict, table_columns: dict, min_iter: int = 0) -> tuple[dict | None, list]:
    """Run one phase of the agent loop. Returns (catalogue_data, last_rejection_reasons)."""
    last_rejection_reasons: list[str] = []
    for i in range(1, max_iter + 1):
        rejection_reasons: list[str] = []
        console.print(f"[dim]  Phase {phase_label} — iteration {i}/{max_iter}…[/dim]")

        response = _call_with_retry(backend, tools)

        for key, val in backend.extract_usage(response).items():
            usage[key] += val

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
                    "time_coverage, data_quality_notes, and description."
                )
                continue
            console.print(f"[yellow]  Agent produced no tool calls in phase {phase_label}[/yellow]")
            return None, last_rejection_reasons

        # Process all tool_use blocks and collect results as (tool_id, content) pairs
        pending_results: list[tuple[str, str]] = []
        catalogue_data = None

        for block in tool_use_blocks:
            if block.name == "run_query":
                result = _execute_query(engine, block.input["sql"], block.input.get("reason", ""))
                _print_query(
                    sql=block.input["sql"],
                    reason=block.input.get("reason", ""),
                    tables=block.input.get("tables", []),
                    columns=block.input.get("columns", []),
                    plain_language=block.input.get("plain_language", ""),
                    result=result,
                    table_columns=table_columns,
                )
                pending_results.append((block.id, result))

            elif block.name == "finish_catalogue":
                # Coerce data_quality_notes to a list of strings — models sometimes
                # submit a count, a single string, or a list of non-strings.
                # If it's not a proper list of strings, discard it so the
                # "empty notes" warning fires and the agent adds real notes.
                dqn = block.input.get("data_quality_notes")
                if not isinstance(dqn, list) or not all(isinstance(n, str) for n in dqn):
                    block.input["data_quality_notes"] = []
                required = {"tables", "measurable_metrics", "queryable_facts", "time_coverage", "description"}
                missing = required - block.input.keys()
                empty_required = [
                    k for k in ("tables", "measurable_metrics")
                    if k in block.input and not block.input[k]
                ]
                rejection_reasons = []
                if block.input.get("_compressed"):
                    rejection_reasons.append(
                        "You submitted a compressed placeholder — this is not a valid catalogue. "
                        "You must submit a completely new finish_catalogue where tables, "
                        "measurable_metrics, and queryable_facts are full objects with all required "
                        "fields (name, description, sql, etc.) — not just a list of names."
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
                    _compress_summary = {
                        "_compressed": True,
                        "description": (block.input.get("description") or "")[:120],
                        "tables": [t.get("name", "?") if isinstance(t, dict) else str(t) for t in (block.input.get("tables") or [])],
                        "measurable_metrics": [m.get("name", "?") if isinstance(m, dict) else str(m) for m in (block.input.get("measurable_metrics") or [])],
                        "queryable_facts": [f.get("name", "?") if isinstance(f, dict) else str(f) for f in (block.input.get("queryable_facts") or [])],
                        "time_coverage": block.input.get("time_coverage"),
                        "data_quality_notes": len(block.input.get("data_quality_notes") or []),
                    }
                    backend.compress_finish_catalogue(block.id, _compress_summary)
                    pending_results.append((
                        block.id,
                        f"ERROR: {reasons_text}. "
                        "Resubmit finish_catalogue with all required fields present and non-empty "
                        "where data was found: tables, measurable_metrics, time_coverage, "
                        "data_quality_notes, description. "
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

        if catalogue_data is not None:
            console.print(f"[green]  finish_catalogue called in phase {phase_label}, iteration {i}.[/green]")
            return catalogue_data, []

        # Track last rejection reasons for the caller to use in the next phase prompt
        if rejection_reasons:
            last_rejection_reasons = rejection_reasons

        # If stop_reason was end_turn despite having tool_use blocks, exit phase
        if backend.stop_reason(response) == "end_turn":
            return None, last_rejection_reasons

    return None, last_rejection_reasons


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


# ── output construction ────────────────────────────────────────────────────────

def _build_catalogue(data: dict, snapshot: dict) -> DataCatalogue:
    try:
        return DataCatalogue(
            analysed_at=datetime.now().isoformat(timespec="seconds"),
            model=MODEL,
            connection=snapshot["connection_string"],
            dialect=snapshot["dialect"],
            description=data.get("description", ""),
            tables=data["tables"],
            measurable_metrics=data["measurable_metrics"],
            queryable_facts=data.get("queryable_facts", []),
            time_coverage=data["time_coverage"],
            data_quality_notes=data.get("data_quality_notes", []),
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


def _write_output(catalogue: DataCatalogue, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(catalogue.model_dump(), f, indent=2)
    console.print(f"\n[bold green]Catalogue written to {out_path}[/bold green]")


# ── phase 3 — refinement ───────────────────────────────────────────────────────

def _run_phase3(
    catalogue: DataCatalogue,
    schema_text: str,
    engine,
    usage: dict,
    table_columns: dict,
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
    _directly_patched = {"date_mismatch", "period_boundary"}
    agent_issues = (
        [
            (m, r) for m, r in warn_metrics
            if set((r.error or "").split(", ")) - _directly_patched
        ] +
        [(f, r) for f, r in warn_facts]
    )

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

    refined_data, _ = _run_phase(
        backend, engine,
        tools=[_RUN_QUERY_TOOL, _FINISH_CATALOGUE_TOOL],
        max_iter=_REFINEMENT_BUDGET,
        phase_label="3 (refinement)",
        usage=usage,
        table_columns=table_columns,
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
    console.print(Panel(
        f"[bold]Schematica[/bold]\n"
        f"[dim]Source:[/dim]  {connection_string}\n"
        f"[dim]Output:[/dim]  {out_path}\n"
        f"[dim]Model:[/dim]   {MODEL}",
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
    args = parser.parse_args()

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
