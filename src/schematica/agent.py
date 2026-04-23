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
import traceback as _traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from rich.panel import Panel
from sqlalchemy import text

from schematica.backends import _AnthropicBackend
from schematica.backends import _LiteLLMBackend
from schematica.catalogue import DataCatalogue
from schematica.catalogue import KeyTerm
from schematica.catalogue import MeasurableMetric
from schematica.catalogue import QueryableFact
from schematica.catalogue import TableRelationship
from schematica.catalogue import TimeRange
from schematica.db import make_readonly_engine
from schematica.db import prompt_readonly_confirmation
from schematica.eval import _is_evaluator_crash
from schematica.eval import evaluate_fact
from schematica.eval import evaluate_metric
from schematica.introspect import introspect
from schematica.introspect import render_as_text
from schematica.output import _PHASE3_WARN_LEGEND
from schematica.output import _calc_rpm
from schematica.output import _format_iter_stats
from schematica.output import _print_finish_catalogue
from schematica.output import _print_header
from schematica.output import _print_query
from schematica.output import _print_schema_detail
from schematica.output import _print_schema_summary
from schematica.output import _print_summary
from schematica.output import _render_overview_md
from schematica.output import _RequestTracker
from schematica.output import console
from schematica.pricing import CACHE_READ_MULTIPLIER
from schematica.pricing import CACHE_WRITE_MULTIPLIER
from schematica.pricing import format_cost
from schematica.pricing import get_context_window as _context_window
from schematica.pricing import get_max_output_tokens as _get_max_output_tokens
from schematica.pricing import get_model_pricing
from sqlalchemy import inspect as sqla_inspect
from schematica.prompts import _BARE_TABLES_ERROR_MSG
from schematica.prompts import _FINISH_CATALOGUE_TOOL
from schematica.prompts import _FK_REJECTION_CAP
from schematica.prompts import _FK_REJECTION_MSG
from schematica.prompts import _TABLES_NOT_LIST_ERROR_MSG
from schematica.prompts import REFINEMENT_SYSTEM_PROMPT
from schematica.prompts import SYSTEM_PROMPT
from schematica.prompts import _update_fk_waived
from schematica.prompts import make_run_query_tool

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if val is None:
        raise RuntimeError(f"{name} is not set. Add it to .env or the environment.")
    return val


def _optional_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _check_package_versions(
    pandas_version: str | None = None,
    numpy_version: str | None = None,
) -> None:
    """Raise RuntimeError if pandas + numpy versions are incompatible.

    pandas < 2.2 uses numpy.rec internally in pd.read_sql(). numpy 2.0 removed
    numpy.rec. The combination silently breaks every Phase 3 eval result.

    Arguments allow injection for testing; production call uses installed versions.
    """
    import numpy
    import pandas

    pv = pandas_version or pandas.__version__
    nv = numpy_version or numpy.__version__

    def _ver(s: str) -> tuple[int, ...]:
        return tuple(int(x) for x in s.split(".")[:2])

    if _ver(pv) < (2, 2) and _ver(nv) >= (2, 0):
        raise RuntimeError(
            f"Incompatible packages: pandas {pv} + numpy {nv}.\n"
            "pandas < 2.2 uses numpy.rec which was removed in numpy 2.0 — "
            "every eval metric will fail with 'No module named numpy.rec'.\n"
            'Fix: uv add "pandas>=2.2" "numpy>=2.0"'
        )


_ANTHROPIC_PREFIX = "anthropic/"


class _ModelConfig:
    """Active model name and caching setting.

    Prompt caching is an Anthropic-only feature — construction fails loudly
    if cache=True is paired with a non-Anthropic model so misconfiguration
    surfaces at startup rather than at the first API call.
    """

    def __init__(self, model: str, cache: bool) -> None:
        if cache and not model.startswith(_ANTHROPIC_PREFIX):
            raise RuntimeError(
                f"Prompt caching requires an '{_ANTHROPIC_PREFIX}' model, got: {model!r}. "
                "Set SC_CACHE=false or change SC_MODEL to e.g. anthropic/claude-haiku-4-5-20251001"
            )
        self.model = model
        self.cache = cache

    @property
    def is_anthropic(self) -> bool:
        return self.model.startswith(_ANTHROPIC_PREFIX)

    @property
    def anthropic_model(self) -> str:
        """Bare model name for the Anthropic SDK (strips the provider prefix)."""
        return self.model[len(_ANTHROPIC_PREFIX):] if self.is_anthropic else self.model


_config = _ModelConfig(
    model=_require_env("SC_MODEL"),
    cache=_optional_env("SC_CACHE", "false").lower() == "true",
)
MAX_ROWS = int(_require_env("SC_MAX_ROWS"))
MAX_CHARS = int(_require_env("SC_MAX_CHARS"))
_BUDGET_BASE = int(_require_env("SC_BUDGET_BASE"))
_BUDGET_MULTIPLIER = int(_require_env("SC_BUDGET_MULTIPLIER"))
_BUDGET_CAP = int(_require_env("SC_BUDGET_CAP"))
_MIN_ITER_FLOOR = int(_require_env("SC_MIN_ITER_FLOOR"))
_MIN_ITER_DIVISOR = int(_require_env("SC_MIN_ITER_DIVISOR"))
_REFINEMENT_BUDGET = int(_require_env("SC_REFINEMENT_BUDGET"))
_MAX_OUTPUT_TOKENS = int(_require_env("SC_MAX_OUTPUT_TOKENS"))
_MAX_QUERIES_PER_TURN = int(_require_env("SC_MAX_QUERIES_PER_TURN"))


def _apply_model_override(new_model: str, cache_override: "bool | None" = None) -> None:
    """Replace _config after CLI --model / --cache flags.

    cache_override=True  → --cache flag was passed; enable caching
    cache_override=None  → no flag; keep current cache setting from .env

    Rules:
      - Non-anthropic model always uses cache=False (caching is Anthropic-only).
        Passing cache_override=True with a non-anthropic model is an error.
      - Anthropic model with cache_override=True overrides the current setting.
      - Anthropic model with no flag keeps the current cache setting.
    """
    global _config
    if not new_model.startswith(_ANTHROPIC_PREFIX):
        if cache_override is True:
            raise RuntimeError(
                f"--cache requires an '{_ANTHROPIC_PREFIX}' model, got: {new_model!r}. "
                "Prompt caching is only supported by Anthropic models."
            )
        new_cache = False
    else:
        new_cache = _config.cache if cache_override is None else bool(cache_override)
    _config = _ModelConfig(new_model, new_cache)


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
    pattern = rf"\b(?:FROM|JOIN)\s+(?:{_IDENT}\s*\.\s*)?{_TABLE}"

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


# ── tool instances (built after MAX_ROWS is resolved from env) ─────────────────
_RUN_QUERY_TOOL = make_run_query_tool(MAX_ROWS)


# ── main entry point ───────────────────────────────────────────────────────────


def run(connection_string: str, out_path: str) -> DataCatalogue:
    """
    Run the Schematica against a database.

    Introspects the schema, runs a two-phase agentic loop to identify and
    validate measurable metrics, then writes the DataCatalogue to out_path.
    """
    _check_package_versions()
    _print_header(connection_string, out_path, _config.model, _config.cache)

    # For SQLite, verify the file exists before spending any tokens.
    if connection_string.startswith("sqlite:///"):
        db_file = connection_string[len("sqlite:///") :]
        if not os.path.exists(db_file):
            console.print(
                f"[bold red]Error:[/bold red] SQLite database not found: {db_file}"
            )
            raise SystemExit(1)

    _probe_connection(make_readonly_engine(connection_string), connection_string)

    console.print("[dim]Introspecting schema…[/dim]")
    snapshot = introspect(connection_string)
    schema_text = render_as_text(snapshot)
    _print_schema_summary(snapshot)
    _print_schema_detail(schema_text)

    table_columns: dict[str, list[str]] = {
        t["name"]: [c["name"] for c in t["columns"]] for t in snapshot["tables"]
    }

    # Collect all FK pairs so _agent_loop can validate that the catalogue
    # contains at least one cross-table metric per FK relationship.
    fk_pairs: list[tuple[str, str]] = [
        (t["name"], fk["to_table"])
        for t in snapshot["tables"]
        for fk in t.get("foreign_keys", [])
    ]

    # Pure junction tables (every column is a FK column, e.g. PlaylistTrack with
    # only PlaylistId and TrackId) have no temporal data and cannot produce a
    # time-series metric on their own — exempt them from metric requirements.
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

    # Lookup tables: no date/time columns → reference/dimension data only.
    # These are exempt from time-series metric requirements and FK cross-metric checks.
    lookup_tables: set[str] = {
        t["name"].lower()
        for t in snapshot["tables"]
        if not _has_temporal_column(t)
    } | junction_tables

    # Temporal tables: have at least one date/time column and are not lookup/junction.
    # Every temporal table must appear in at least one measurable_metric.
    required_tables: set[str] = {
        t["name"].lower()
        for t in snapshot["tables"]
        if _has_temporal_column(t) and t["name"].lower() not in junction_tables
    }

    engine = make_readonly_engine(connection_string)
    n_tables = len(snapshot["tables"])
    budget = _phase1_budget(n_tables)
    min_iter = max(_MIN_ITER_FLOOR, budget // _MIN_ITER_DIVISOR)
    console.print(
        f"[dim]Exploration budget: {budget} iterations for {n_tables} tables  (min {min_iter} before finish)[/dim]\n"
    )

    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "thinking_tokens": 0,
        "total_cost": 0.0,
    }
    started_at = time.monotonic()
    req_tracker = _RequestTracker(started_at)
    try:
        catalogue_data = _agent_loop(
            schema_text,
            engine,
            budget,
            min_iter,
            usage,
            table_columns,
            fk_pairs,
            lookup_tables,
            started_at,
            req_tracker,
            required_tables=required_tables,
        )
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        console.print(f"[dim]{_traceback.format_exc()}[/dim]")
        catalogue_data = None
    elapsed_secs = time.monotonic() - started_at

    if catalogue_data is None:
        inp = usage["input_tokens"]
        out = usage["output_tokens"]
        cache_created = usage.get("cache_creation_tokens", 0)
        cache_read = usage.get("cache_read_tokens", 0)
        mins, secs = divmod(int(elapsed_secs), 60)
        elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        console.print(
            f"[bold red]Agent failed to submit a valid catalogue after exhausting all retries.[/bold red]\n"
            "The agent either ran out of budget or repeatedly submitted incomplete finish_catalogue calls.\n"
            "No output file written. Re-run with a larger budget or inspect the agent log above.\n"
            f"[dim]Tokens: {inp:,} in + {out:,} out"
            + (
                f" + {cache_created:,} cache write + {cache_read:,} cache read"
                if cache_created or cache_read
                else ""
            )
            + f"  |  Cost: {format_cost(_config.model, inp, out, cache_created, cache_read)}"
            + f"  |  Elapsed: {elapsed_str}[/dim]"
        )
        raise SystemExit(1)

    catalogue = _build_catalogue(catalogue_data, snapshot)
    catalogue = _drop_broken_sql(catalogue, engine)
    catalogue, final_metric_results, final_fact_results, uncovered_tables = (
        _run_phase3_safe(
            _run_phase3,
            catalogue,
            schema_text,
            engine,
            usage,
            table_columns,
            tracker=req_tracker,
        )
    )
    _write_output(catalogue, out_path)
    _print_summary(
        catalogue,
        usage,
        elapsed_secs,
        final_metric_results,
        final_fact_results,
        uncovered_tables,
        _config.model,
    )

    return catalogue


def _pre_validate_catalogue_items(data: dict) -> list[str]:
    """Pre-validate per-item fields in a finish_catalogue submission.

    Returns a list of human-readable error strings. An empty list means all
    items passed validation. Called before accepting the submission so errors
    are returned to the agent as rejection feedback rather than crashing later
    in _build_catalogue.
    """
    errors: list[str] = []

    for idx, m in enumerate(data.get("measurable_metrics", [])):
        if not isinstance(m, dict):
            continue
        try:
            MeasurableMetric.model_validate(m)
        except Exception as e:
            errors.append(f"measurable_metrics[{idx}] ({m.get('name', '?')}): {e}")

    for idx, f in enumerate(data.get("queryable_facts", [])):
        if not isinstance(f, dict):
            continue
        try:
            QueryableFact.model_validate(f)
        except Exception as e:
            errors.append(f"queryable_facts[{idx}] ({f.get('name', '?')}): {e}")

    for idx, k in enumerate(data.get("key_terms", [])):
        if not isinstance(k, dict):
            continue
        try:
            KeyTerm.model_validate(k)
        except Exception as e:
            errors.append(f"key_terms[{idx}] ({k.get('term', '?')}): {e}")

    for idx, r in enumerate(data.get("table_relationships", [])):
        if not isinstance(r, dict):
            continue
        try:
            TableRelationship.model_validate(r)
        except Exception as e:
            errors.append(f"table_relationships[{idx}]: {e}")

    return errors


def _compact_json(obj) -> str:
    """Serialize *obj* to JSON without indentation for use in LLM prompts.

    Compact serialization keeps token counts low. Use indent=2 only when writing
    to disk or displaying to the user.
    """
    return json.dumps(obj, separators=(",", ":"))


def _debug_dump_conversation(backend) -> None:
    """Print the full conversation history to stderr for debugging empty-response failures.

    TODO: remove once the Gemini empty-response root cause is identified.
    """
    import sys
    sep = "=" * 80
    print(f"\n{sep}\nDEBUG CONVERSATION DUMP\n{sep}", file=sys.stderr)
    system = getattr(backend, "_system", "<unavailable>")
    messages = getattr(backend, "messages", [])
    print(f"[SYSTEM]\n{system}\n", file=sys.stderr)
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        parts: list[str] = []

        # ── text content (string or Anthropic content-block list) ─────────────
        content = msg.get("content", "")
        if isinstance(content, str):
            if content:
                parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    parts.append(str(block)[:300])
                    continue
                btype = block.get("type", "?")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    # Anthropic-format tool call
                    raw = json.dumps(block.get("input", {}))
                    parts.append(f"[tool_use: {block.get('name')} input={raw[:300]}]")
                elif btype == "tool_result":
                    c = block.get("content", "")
                    if isinstance(c, list):
                        c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
                    parts.append(f"[tool_result id={block.get('tool_use_id')} {str(c)[:300]}]")
                else:
                    parts.append(f"[{btype}]")

        # ── LiteLLM/OpenAI-format tool calls (separate field on assistant msg) ─
        for tc in msg.get("tool_calls", []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {})
            name = fn.get("name", "?")
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except (json.JSONDecodeError, TypeError):
                args = args_raw
            # Show the most useful fields for run_query / finish_catalogue
            if name == "run_query":
                sql = str(args.get("sql", ""))[:200]
                reason = args.get("reason", "")
                parts.append(f"[tool_call: run_query sql={sql!r} reason={reason!r}]")
            elif name == "finish_catalogue":
                n_metrics = len(args.get("measurable_metrics", []))
                n_facts = len(args.get("queryable_facts", []))
                parts.append(
                    f"[tool_call: finish_catalogue metrics={n_metrics} facts={n_facts}]"
                )
            else:
                parts.append(f"[tool_call: {name} args={json.dumps(args)[:200]}]")

        body = "\n".join(parts) if parts else "<empty>"
        print(f"[{i}] {role.upper()}\n{body}\n", file=sys.stderr)
    print(sep, file=sys.stderr)


def _call_with_retry(
    backend, tools: list, max_tokens: int = _MAX_OUTPUT_TOKENS, max_attempts: int = 7,
):
    """Call backend.call() with backoff on rate-limit errors.

    Uses the retry-after header hint from the exception when available (exact
    wait the API requested).  Falls back to exponential backoff otherwise.

    Empty-choices responses are re-raised immediately so the caller (_run_phase)
    can surface them as their own numbered iteration before retrying.
    """
    delay = (
        65  # starting own-backoff; 65s clears a 60s rate-limit window on first retry
    )
    for attempt in range(1, max_attempts + 1):
        try:
            return backend.call(tools, max_tokens)
        except Exception as exc:
            msg = str(exc).lower()
            is_unsupported_tools = (
                "does not support parameters" in msg and "tools" in msg
            ) or ("unsupportedparams" in msg and "tools" in msg)
            if is_unsupported_tools:
                raise RuntimeError(
                    f"Model does not support tool calls, which schematica requires. "
                    "Choose a model with function/tool calling support "
                    "(e.g. gemini/gemini-2.5-flash, anthropic/claude-sonnet-4-6)."
                ) from exc
            if "empty choices" in msg:
                raise  # handled by _run_phase as a new iteration
            is_rate_limit = (
                "rate limit" in msg
                or "ratelimit" in msg
                or "429" in msg
                or "rate_limited" in msg
            )
            is_transient = "overloaded" in msg or "503" in msg or "502" in msg
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


def _make_backend(
    initial_user_text: str, system_prompt: str, client=None
) -> "_AnthropicBackend | _LiteLLMBackend":
    """Create the right backend for the configured provider.

    For the Anthropic path, an optional pre-created *client* can be supplied so
    that multiple backends (e.g. successive ensemble Phase-1 runs) share a single
    SDK client object instead of constructing a new one each time.  Each backend
    still gets its own fresh *messages* list so the conversation histories are
    independent.
    """
    if _config.is_anthropic:
        if client is None:
            client = _get_anthropic_client()
        if _config.cache:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": initial_user_text,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": initial_user_text}]
        return _AnthropicBackend(client, _config.anthropic_model, system_prompt, messages, cache=_config.cache)
    else:
        messages = [{"role": "user", "content": initial_user_text}]
        return _LiteLLMBackend(_config.model, system_prompt, messages)


# ── agent loop ─────────────────────────────────────────────────────────────────


_ENSEMBLE_RESULT_TRUNCATE = 800  # chars per query result in the combined context

# Desired output-token budget for Phase-2/3 (documentation / refinement).
# We want to give the agent as much room as possible to write a large catalogue,
# but must not exceed the model's hard per-response limit.
_PHASE2_TARGET_TOKENS = 65_536


def _phase2_max_tokens() -> int:
    """Phase-2/3 output token budget, capped by the active model's hard limit."""
    return min(_PHASE2_TARGET_TOKENS, _get_max_output_tokens(_config.model))


def _dedup_query_logs(query_logs: list[list[dict]]) -> list[list[dict]]:
    """Remove duplicate SQL queries across ensemble runs.

    When N Phase-1 runs explore the same database they often converge on the
    same validation queries (e.g. MIN/MAX date range, row counts).  Sending
    identical SQL blocks to Phase 2 inflates the context without adding
    information.  We keep the *first* occurrence of each SQL string and drop
    later duplicates, preserving per-run grouping so the run-section headers
    in the Phase-2 prompt remain accurate.
    """
    seen_sql: set[str] = set()
    deduped: list[list[dict]] = []
    for log in query_logs:
        run_deduped: list[dict] = []
        for q in log:
            sql_key = q["sql"].strip()
            if sql_key not in seen_sql:
                seen_sql.add(sql_key)
                run_deduped.append(q)
        deduped.append(run_deduped)
    return deduped


def _format_ensemble_context(schema_text: str, query_logs: list[list[dict]]) -> str:
    """Format N Phase-1 query logs into a single Phase-2 initial message.

    Each entry in query_logs is the query_log returned by one Phase-1 _run_phase
    call: a list of dicts with keys sql, reason, plain_language, result.

    The returned string provides the schema snapshot and every validated
    query + result so the Phase-2 agent has the full exploration evidence.
    """
    n = len(query_logs)
    total_queries = sum(len(log) for log in query_logs)
    parts: list[str] = [
        f"Here is the complete schema snapshot of the database:\n\n"
        f"```\n{schema_text}\n```\n\n"
        f"PHASE 1 was run {n} time{'s' if n != 1 else ''} independently on this database. "
        f"Each run explored the schema and validated SQL against the live data. "
        f"All {total_queries} validated queries and their results are provided below.\n"
        f"Use this combined evidence to produce the most comprehensive and accurate catalogue possible.\n",
    ]
    bar = "═" * 60
    for run_idx, log in enumerate(query_logs, 1):
        parts.append(f"\n{bar}\nEXPLORATION RUN {run_idx} / {n}  ({len(log)} queries)\n{bar}\n")
        for q_idx, q in enumerate(log, 1):
            label = q.get("plain_language") or q.get("reason") or f"Query {q_idx}"
            result = q["result"]
            if len(result) > _ENSEMBLE_RESULT_TRUNCATE:
                result = result[:_ENSEMBLE_RESULT_TRUNCATE] + "\n... (truncated)"
            parts.append(
                f"\n[{q_idx}] {label}\n"
                f"SQL: {q['sql']}\n"
                f"Result:\n{result}\n"
            )
    parts.append(
        f"\n{bar}\n"
        "You are in PHASE 2 — DOCUMENTATION. The run_query tool is not available.\n"
        "Call finish_catalogue now. Draw on all exploration runs above — include every\n"
        "metric that was validated by at least one run, and every table that was queried."
    )
    return "".join(parts)


def _agent_loop_ensemble(
    schema_text: str,
    engine,
    phase1_budget: int,
    phase1_min_iter: int,
    n_ensemble: int,
    usage: dict,
    table_columns: dict,
    fk_pairs: list[tuple[str, str]] | None,
    lookup_tables: set[str] | None,
    started_at: float,
    tracker: "_RequestTracker | None",
    required_tables: set[str] | None,
    min_metrics: int,
    initial_message: str,
) -> dict:
    """Run N independent Phase-1 explorations, then one fresh Phase-2 documentation pass.

    Each Phase-1 run gets its own backend and conversation history.  Their
    validated query logs are combined into a single context for Phase-2 so
    the documentation agent sees the full exploration evidence from all runs.

    All ensemble backends share a single underlying API client object so we
    avoid repeated SSL/transport initialisation on every run.
    """
    console.print(
        Panel(
            f"[bold cyan]Ensemble mode — {n_ensemble} independent Phase-1 explorations[/bold cyan]\n"
            "[dim]Each run explores and validates SQL independently. "
            "All results are combined for a single Phase-2 documentation pass.[/dim]",
            border_style="cyan",
            padding=(0, 1),
        )
    )

    # Create one shared SDK client for ALL backends in this ensemble run.
    # The Anthropic/LiteLLM clients are stateless HTTP wrappers — only the
    # messages list (conversation history) differs per run.
    _shared_client = _get_anthropic_client() if _config.is_anthropic else None

    all_query_logs: list[list[dict]] = []
    # Keep any finish_catalogue payload submitted during Phase 1 as a fallback in
    # case Phase 2 fails (e.g. safety filter on the combined context).
    phase1_fallback: dict | None = None

    for run_num in range(1, n_ensemble + 1):
        console.print(
            Panel(
                f"[bold]Ensemble exploration {run_num} / {n_ensemble}[/bold]",
                border_style="blue",
                padding=(0, 1),
            )
        )
        run_backend = _make_backend(initial_message, SYSTEM_PROMPT, client=_shared_client)
        catalogue_data, _, _, query_log = _run_phase(
            run_backend,
            engine,
            tools=[_RUN_QUERY_TOOL, _FINISH_CATALOGUE_TOOL],
            max_iter=phase1_budget,
            min_iter=phase1_min_iter,
            phase_label=f"1.{run_num}/{n_ensemble} (exploration)",
            usage=usage,
            table_columns=table_columns,
            fk_pairs=fk_pairs,
            lookup_tables=lookup_tables,
            tracker=tracker,
            required_tables=required_tables,
            min_metrics=min_metrics,
        )
        if catalogue_data is not None:
            phase1_fallback = catalogue_data  # most recent valid Phase-1 result
        all_query_logs.append(query_log)
        console.print(
            f"[dim]  Exploration {run_num}/{n_ensemble} complete — "
            f"{len(query_log)} queries validated[/dim]\n"
        )

    total_q = sum(len(l) for l in all_query_logs)
    # Deduplicate queries by SQL so runs that explored the same query don't
    # bloat the Phase-2 context with identical entries.
    deduped_logs = _dedup_query_logs(all_query_logs)
    deduped_q = sum(len(l) for l in deduped_logs)
    dedup_note = (
        f" ({total_q - deduped_q} duplicate SQL queries removed)" if deduped_q < total_q else ""
    )
    console.print(
        Panel(
            f"[bold cyan]All {n_ensemble} explorations complete — entering Phase 2 (documentation)[/bold cyan]\n"
            f"[dim]{deduped_q} unique validated queries across {n_ensemble} runs{dedup_note}. "
            "Compiling combined catalogue.[/dim]",
            border_style="cyan",
            padding=(0, 1),
        )
    )

    ensemble_context = _format_ensemble_context(schema_text, deduped_logs)
    phase2_backend = _make_backend(ensemble_context, SYSTEM_PROMPT, client=_shared_client)

    try:
        catalogue_data, _, _, _ = _run_phase(
            phase2_backend,
            engine,
            tools=[_FINISH_CATALOGUE_TOOL],
            max_iter=5,
            phase_label="2 (ensemble documentation)",
            usage=usage,
            table_columns=table_columns,
            required_tables=required_tables,
            min_metrics=min_metrics,
            tracker=tracker,
            initial_max_tokens=_phase2_max_tokens(),
        )
    except RuntimeError as exc:
        if phase1_fallback is not None:
            console.print(
                f"[yellow]  Phase 2 failed ({exc}). "
                "Falling back to best Phase-1 catalogue.[/yellow]"
            )
            return phase1_fallback
        raise

    if catalogue_data is not None:
        return catalogue_data

    if phase1_fallback is not None:
        console.print(
            "[yellow]  Phase 2 produced no catalogue. "
            "Falling back to best Phase-1 catalogue.[/yellow]"
        )
        return phase1_fallback

    raise RuntimeError(
        f"Ensemble agent did not produce a catalogue after {n_ensemble} explorations."
    )


def _agent_loop(
    schema_text: str,
    engine,
    phase1_budget: int,
    phase1_min_iter: int,
    usage: dict,
    table_columns: dict,
    fk_pairs: list[tuple[str, str]] | None = None,
    lookup_tables: set[str] | None = None,
    started_at: float = 0.0,
    tracker: "_RequestTracker | None" = None,
    required_tables: set[str] | None = None,
    min_metrics: int = 0,
) -> dict:
    initial_message = (
        f"Here is the complete schema snapshot of the database:\n\n"
        f"```\n{schema_text}\n```\n\n"
        f"You are in PHASE 1 — EXPLORATION. You have {phase1_budget} iterations "
        f"to explore the schema and validate your SQL queries. "
        f"Start by identifying what each table represents, then run queries to "
        f"validate the metrics you plan to include in the catalogue."
    )

    n_ensemble = int(_optional_env("SC_ENSEMBLE_RUNS", "1"))

    if n_ensemble > 1:
        return _agent_loop_ensemble(
            schema_text=schema_text,
            engine=engine,
            phase1_budget=phase1_budget,
            phase1_min_iter=phase1_min_iter,
            n_ensemble=n_ensemble,
            usage=usage,
            table_columns=table_columns,
            fk_pairs=fk_pairs,
            lookup_tables=lookup_tables,
            started_at=started_at,
            tracker=tracker,
            required_tables=required_tables,
            min_metrics=min_metrics,
            initial_message=initial_message,
        )

    # ── Single-run path (n_ensemble == 1) — original behaviour ─────────────────
    # Phase 1 and Phase 2 share the same backend — conversation history carries forward.
    backend = _make_backend(initial_message, SYSTEM_PROMPT)
    catalogue_data, last_rejection_reasons, _, _ = _run_phase(
        backend,
        engine,
        tools=[_RUN_QUERY_TOOL, _FINISH_CATALOGUE_TOOL],
        max_iter=phase1_budget,
        min_iter=phase1_min_iter,
        phase_label="1 (exploration)",
        usage=usage,
        table_columns=table_columns,
        fk_pairs=fk_pairs,
        lookup_tables=lookup_tables,
        tracker=tracker,
        required_tables=required_tables,
        min_metrics=min_metrics,
    )
    if catalogue_data is not None:
        return catalogue_data

    # Phase 2 — documentation, only finish_catalogue available
    console.print(
        Panel(
            "[bold cyan]Phase 1 complete — entering Phase 2 (documentation)[/bold cyan]\n"
            "[dim]run_query is no longer available. The agent must now compile and submit finish_catalogue.[/dim]",
            border_style="cyan",
            padding=(0, 1),
        )
    )

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

    catalogue_data, _, _, _ = _run_phase(
        backend,
        engine,
        tools=[_FINISH_CATALOGUE_TOOL],
        max_iter=5,
        phase_label="2 (documentation)",
        usage=usage,
        table_columns=table_columns,
        required_tables=required_tables,
        min_metrics=min_metrics,
        tracker=tracker,
        initial_max_tokens=_phase2_max_tokens(),
    )
    if catalogue_data is not None:
        return catalogue_data

    raise RuntimeError("Agent did not produce a catalogue after both phases.")


def _run_phase(
    backend,
    engine,
    tools: list,
    max_iter: int,
    phase_label: str,
    usage: dict,
    table_columns: dict,
    min_iter: int = 0,
    fk_pairs: list[tuple[str, str]] | None = None,
    lookup_tables: set[str] | None = None,
    tracker: "_RequestTracker | None" = None,
    nudge_text: str | None = None,
    initial_max_tokens: int = 0,
    required_tables: set[str] | None = None,
    min_metrics: int = 0,
) -> tuple[dict | None, list, int, list[dict]]:
    """Run one phase of the agent loop.

    Returns (catalogue_data, last_rejection_reasons, iterations_run, query_log).
    query_log is a list of every run_query call that executed: each entry has
    keys sql, reason, plain_language, result.
    """

    # Proactive output-token throttling: only for the Anthropic backend, where
    # the per-minute limit is available from response headers.
    output_bucket: _OutputTokenBucket | None = (
        _OutputTokenBucket() if isinstance(backend, _AnthropicBackend) else None
    )
    last_out_tokens: int = 0  # previous iteration's output; used as estimate

    last_rejection_reasons: list[str] = []
    fk_rejection_counts: dict = {}
    fk_waived: set = set()
    query_log: list[dict] = []
    _current_max_tokens: int = initial_max_tokens if initial_max_tokens > 0 else _MAX_OUTPUT_TOKENS
    _consecutive_empty: int = 0
    _best_effort_catalogue: dict | None = None  # most recent rejected finish_catalogue, used as last-resort fallback
    for i in range(1, max_iter + 1):
        rejection_reasons: list[str] = []
        console.rule(f"[dim] Phase {phase_label} — iter {i}/{max_iter} [/dim]", style="dim", characters="━")

        call_start = time.monotonic()
        if output_bucket is not None:
            waited = output_bucket.proactive_wait(
                now=call_start, expected=last_out_tokens
            )
            if waited > 0:
                console.print(
                    f"[yellow]  Output token budget near limit — waited {waited:.1f}s "
                    f"(proactive throttle)[/yellow]"
                )

        try:
            response = _call_with_retry(backend, tools, max_tokens=_current_max_tokens)
            _consecutive_empty = 0
        except ValueError as exc:
            if "empty choices" not in str(exc).lower():
                raise
            _consecutive_empty += 1
            _et = getattr(exc, "empty_response_tokens", {})
            # Account for tokens consumed by the empty response — always,
            # even if this is the second consecutive empty and we're about to raise.
            _empty_now = time.monotonic()
            _empty_duration = _empty_now - call_start
            if tracker is not None:
                tracker.record(now=_empty_now)
            _empty_in  = _et.get("prompt_tokens")   or 0
            _empty_out = _et.get("output_tokens")    or 0
            _empty_think = _et.get("thinking_tokens") or 0
            usage["input_tokens"]   += _empty_in
            usage["output_tokens"]  += _empty_out
            usage["thinking_tokens"] += _empty_think
            _ep = get_model_pricing(_config.model)
            _empty_cost = (_empty_in * _ep["input"] + _empty_out * _ep["output"]) / 1_000_000
            usage["total_cost"] += _empty_cost
            _empty_stats = _format_iter_stats(
                _empty_in, _empty_out, _config.model,
                iter_duration=_empty_duration,
                iter_num=i,
                max_iter=max_iter,
                context_window=_context_window(_config.model),
                thinking_tokens=_empty_think,
                all_iters=tracker.total if tracker is not None else i,
                session_tracker=tracker,
                now=_empty_now,
                session_total_in=usage.get("input_tokens", 0),
                session_total_out=usage.get("output_tokens", 0),
                session_total_thinking=usage.get("thinking_tokens", 0),
                session_total_cost=usage["total_cost"],
                session_total_cache_create=usage.get("cache_creation_tokens", 0),
                session_total_cache_read=usage.get("cache_read_tokens", 0),
            )
            for _line in _empty_stats.splitlines():
                console.print(f"[dim]  {_line}[/dim]")
            console.print()
            if _consecutive_empty >= 3:
                if _best_effort_catalogue is not None:
                    console.print(
                        "[yellow]  3 consecutive empty responses — falling back to best-effort "
                        "catalogue from earlier 'too early' submission.[/yellow]"
                    )
                    return _best_effort_catalogue, last_rejection_reasons, i, query_log
                raise RuntimeError(
                    "Model returned an empty response 3 times in a row "
                    "(empty choices — API content-policy filter). "
                    "The conversation context may have triggered a safety filter. "
                    "Try a different model or database."
                ) from exc
            # Double the output budget before retrying. Gemini 2.5 Flash's
            # dynamic thinking can consume a large share of max_tokens,
            # leaving too little for actual output.
            _current_max_tokens = min(_current_max_tokens * 2, _phase2_max_tokens())
            console.print(
                f"[yellow]  Empty response from API — nudging and retrying "
                f"(output budget raised to {_current_max_tokens:,} tokens)[/yellow]"
            )
            # Only nudge if there are iterations left — on the last iteration the
            # phase is ending and a dangling nudge would stack with the next phase's
            # transition message, confusing the model with consecutive user turns.
            if i < max_iter:
                default_nudge = (
                    "Your previous response was empty — the API returned no content. "
                    "Please continue: call run_query to explore the database further, "
                    "or call finish_catalogue if you have enough data."
                )
                backend.append_user(nudge_text if nudge_text is not None else default_nudge)
            continue  # this increments i, making the retry a new numbered iteration
        now = time.monotonic()
        iter_duration = now - call_start

        if tracker is not None:
            tracker.record(now=now)

        iter_usage = backend.extract_usage(response)
        for key, val in iter_usage.items():
            usage[key] += val

        iter_in = iter_usage.get("input_tokens", 0)
        iter_out = iter_usage.get("output_tokens", 0)
        iter_cache_create = iter_usage.get("cache_creation_tokens", 0)
        iter_cache_read = iter_usage.get("cache_read_tokens", 0)
        iter_thinking = iter_usage.get("thinking_tokens", 0)

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
        _p = get_model_pricing(_config.model)

        iter_cost = (iter_in * _p["input"] + iter_out * _p["output"]) / 1_000_000
        if iter_cache_create:
            iter_cost += (
                iter_cache_create
                * _p.get("cache_write", _p["input"] * CACHE_WRITE_MULTIPLIER)
                / 1_000_000
            )
        if iter_cache_read:
            iter_cost += (
                iter_cache_read
                * _p.get("cache_read", _p["input"] * CACHE_READ_MULTIPLIER)
                / 1_000_000
            )
        usage["total_cost"] += iter_cost

        _iter_stats = _format_iter_stats(
            iter_in,
            iter_out,
            _config.model,
            iter_duration=iter_duration,
            iter_num=i,
            max_iter=max_iter,
            context_window=_context_window(_config.model),
            cache_creation_tokens=iter_cache_create,
            cache_read_tokens=iter_cache_read,
            thinking_tokens=iter_thinking,
            all_iters=tracker.total if tracker is not None else i,
            session_tracker=tracker,
            now=now,
            session_total_in=usage.get("input_tokens", 0),
            session_total_out=usage.get("output_tokens", 0),
            session_total_thinking=usage.get("thinking_tokens", 0),
            session_total_cost=usage["total_cost"],
            session_total_cache_create=usage.get("cache_creation_tokens", 0),
            session_total_cache_read=usage.get("cache_read_tokens", 0),
        )

        def _print_iter_stats() -> None:
            for line in _iter_stats.splitlines():
                console.print(f"[dim]  {line}[/dim]")
            console.print()

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
            _print_iter_stats()
            continue

        # Collect any tool_use blocks regardless of stop_reason
        tool_use_blocks = backend.tool_calls(response)

        if not tool_use_blocks:
            if i < min_iter:
                # Still in the mandatory exploration window — nudge to keep exploring
                console.print(
                    f"[yellow]  No tool calls in phase {phase_label} iteration {i} — nudging to continue[/yellow]"
                )
                backend.append_user(
                    "You must keep exploring. Call run_query to investigate more tables and "
                    "cross-table JOIN opportunities before calling finish_catalogue. "
                    f"You have used {i}/{max_iter} iterations; minimum required is {min_iter}."
                )
                _print_iter_stats()
                continue
            if i < max_iter:
                # Past the minimum but haven't submitted yet — nudge to finish
                console.print(
                    f"[yellow]  No tool calls in phase {phase_label} iteration {i} — nudging to submit[/yellow]"
                )
                backend.append_user(
                    "You have explored enough. Call finish_catalogue now with everything "
                    "you have learned. Include all tables, measurable_metrics, queryable_facts, "
                    "time_coverage, data_quality_notes, description, and overview."
                )
                _print_iter_stats()
                continue
            console.print(
                f"[yellow]  Agent produced no tool calls in phase {phase_label}[/yellow]"
            )
            _print_iter_stats()
            return None, last_rejection_reasons, i, query_log

        # Process all tool_use blocks and collect results as (tool_id, content) pairs
        pending_results: list[tuple[str, str]] = []
        catalogue_data = None

        # Run all run_query blocks concurrently, then print results in order.
        # executor.map preserves input order so zip is safe.
        run_query_blocks = [b for b in tool_use_blocks if b.name == "run_query"]
        if _MAX_QUERIES_PER_TURN > 0 and len(run_query_blocks) > _MAX_QUERIES_PER_TURN:
            deferred_blocks = run_query_blocks[_MAX_QUERIES_PER_TURN:]
            run_query_blocks = run_query_blocks[:_MAX_QUERIES_PER_TURN]
            for b in deferred_blocks:
                pending_results.append((b.id, "Query deferred — too many queries in one turn. Resubmit in your next response."))
        for block, (block_id, result) in zip(
            run_query_blocks, _run_queries_parallel(engine, run_query_blocks)
        ):
            query_log.append({
                "sql": block.input["sql"],
                "reason": block.input.get("reason", ""),
                "plain_language": block.input.get("plain_language", ""),
                "result": result,
            })
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
                if not isinstance(dqn, list) or not all(
                    isinstance(n, str) for n in dqn
                ):
                    block.input["data_quality_notes"] = []
                required = {
                    "tables",
                    "measurable_metrics",
                    "queryable_facts",
                    "description",
                    "overview",
                }
                missing = required - block.input.keys()
                empty_required = [
                    k
                    for k in ("tables", "measurable_metrics")
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
                        f"tables[0] is {raw_tables[0]!r} (a string). "
                        + _BARE_TABLES_ERROR_MSG
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
                all_items = list(block.input.get("measurable_metrics", [])) + list(
                    block.input.get("queryable_facts", [])
                )
                tables_used_errors = _tables_used_violations(all_items)
                if tables_used_errors:
                    rejection_reasons.append(
                        "tables_used mismatch — tables_used must list only tables that appear "
                        "in the SQL FROM/JOIN clauses:\n  "
                        + "\n  ".join(tables_used_errors)
                    )
                # FK coverage: every FK relationship must have at least one metric
                # whose SQL JOINs both tables. Pairs that have been rejected
                # _FK_REJECTION_CAP times are waived to prevent infinite loops.
                if fk_pairs:
                    submitted_metrics = [
                        m
                        for m in block.input.get("measurable_metrics", [])
                        if isinstance(m, dict)
                    ]
                    effective_fk_pairs = [
                        p for p in fk_pairs if frozenset(p) not in fk_waived
                    ]
                    missing_fks = _uncovered_fk_pairs(
                        submitted_metrics, effective_fk_pairs, lookup_tables
                    )
                    if missing_fks:
                        fk_rejection_counts, fk_waived = _update_fk_waived(
                            missing_fks, fk_rejection_counts, fk_waived
                        )
                        pairs_str = ", ".join(f"{a}↔{b}" for a, b in missing_fks)
                        rejection_reasons.append(
                            _FK_REJECTION_MSG.format(pairs_str=pairs_str)
                        )
                # Per-table coverage: every non-lookup table with temporal data
                # must appear in at least one measurable_metric.
                if required_tables and not rejection_reasons:
                    submitted_metrics = submitted_metrics if fk_pairs else [
                        m for m in block.input.get("measurable_metrics", [])
                        if isinstance(m, dict)
                    ]
                    covered = {
                        t.lower()
                        for m in submitted_metrics
                        for t in m.get("tables_used", [])
                    }
                    missing_coverage = sorted(
                        t for t in required_tables if t.lower() not in covered
                    )
                    if missing_coverage:
                        rejection_reasons.append(
                            f"Missing measurable_metrics for tables with temporal data: "
                            f"{', '.join(missing_coverage)}. Each of these tables must appear "
                            f"in at least one measurable_metric's tables_used."
                        )
                # Metric count floor: guard against sparse runs.
                if min_metrics > 0 and not rejection_reasons:
                    n_submitted = len([
                        m for m in block.input.get("measurable_metrics", [])
                        if isinstance(m, dict)
                    ])
                    if n_submitted < min_metrics:
                        rejection_reasons.append(
                            f"Too few metrics: {n_submitted} submitted, minimum is {min_metrics}. "
                            f"Explore more of the schema and add measurable_metrics for all "
                            f"tables that contain temporal data."
                        )
                # Pre-validate all per-item fields so errors are returned to the
                # agent as rejection feedback rather than crashing in _build_catalogue.
                if not rejection_reasons:
                    schema_errors = _pre_validate_catalogue_items(block.input)
                    if schema_errors:
                        rejection_reasons.append(
                            "Schema validation errors — fix these field values:\n  "
                            + "\n  ".join(schema_errors)
                        )
                _print_finish_catalogue(
                    block.input, accepted=not bool(rejection_reasons)
                )
                if rejection_reasons:
                    reasons_text = "; ".join(rejection_reasons)
                    console.print(
                        f"[yellow]  finish_catalogue rejected — {reasons_text}[/yellow]"
                    )
                    # Replace the full catalogue payload in the assistant message with a
                    # compact summary so it doesn't bloat the context on every subsequent
                    # iteration. The tool_result feedback is sufficient for the agent to
                    # know what was missing.
                    _raw_tables = block.input.get("tables")
                    _raw_metrics = block.input.get("measurable_metrics")
                    _raw_facts = block.input.get("queryable_facts")
                    _compress_summary = {
                        "_compressed": True,
                        "description": (block.input.get("description") or "")[:120],
                        "tables": (
                            [
                                t.get("name", "?") if isinstance(t, dict) else str(t)
                                for t in _raw_tables
                            ]
                            if isinstance(_raw_tables, list)
                            else []
                        ),
                        "measurable_metrics": (
                            [
                                m.get("name", "?") if isinstance(m, dict) else str(m)
                                for m in _raw_metrics
                            ]
                            if isinstance(_raw_metrics, list)
                            else []
                        ),
                        "queryable_facts": (
                            [
                                f.get("name", "?") if isinstance(f, dict) else str(f)
                                for f in _raw_facts
                            ]
                            if isinstance(_raw_facts, list)
                            else []
                        ),
                        "data_quality_notes": len(
                            block.input.get("data_quality_notes") or []
                        ),
                    }
                    # Always save the most recent submission as a best-effort fallback.
                    # If the model later gets stuck returning empty responses, we use
                    # this rather than failing with no output at all.
                    _best_effort_catalogue = block.input
                    backend.compress_finish_catalogue(block.id, _compress_summary)
                    pending_results.append(
                        (
                            block.id,
                            f"ERROR: {reasons_text}. "
                            "Resubmit finish_catalogue with all required fields present and non-empty "
                            "where data was found: tables, measurable_metrics, description, overview. "
                            "queryable_facts may be empty if none were found. "
                            "time_coverage, table_relationships, and per-table data_quality_notes "
                            "are pre-filled — do not include them.",
                        )
                    )
                else:
                    catalogue_data = block.input
                    pending_results.append((block.id, "Catalogue accepted."))

        # All tool output is done — print stats before state management
        _print_iter_stats()

        # Always append tool_results to keep the message history valid
        backend.append_tool_results(pending_results)
        backend.compress_old_run_queries()
        last_out_tokens = iter_out

        if catalogue_data is not None:
            console.print(
                f"[green]  finish_catalogue called in phase {phase_label}, iteration {i}.[/green]"
            )
            return catalogue_data, [], i, query_log

        # Track last rejection reasons for the caller to use in the next phase prompt
        if rejection_reasons:
            last_rejection_reasons = rejection_reasons

        # If stop_reason was end_turn despite having tool_use blocks, exit phase
        if backend.stop_reason(response) == "end_turn":
            return None, last_rejection_reasons, i, query_log

    return None, last_rejection_reasons, max_iter, query_log


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
        result = _execute_query(
            engine, block.input["sql"], block.input.get("reason", "")
        )
        return (block.id, result)

    with ThreadPoolExecutor(max_workers=len(run_query_blocks)) as executor:
        return list(executor.map(_run, run_query_blocks))


# ── output construction ────────────────────────────────────────────────────────

_NULL_RATE_THRESHOLD = 0.20  # columns with >= 20% nulls get a data quality note
_SMALL_TABLE_ROWS    = 100   # tables with fewer rows than this get a size note

# Date detection — used by _has_temporal_column and _deterministic_catalogue_fields.
# Matches column type names (works for typed DBs) OR ISO-date-shaped min/max values
# (SQLite stores dates as TEXT, so type matching alone misses them).
_DATE_TYPE_HINTS = ("DATE", "TIME", "TIMESTAMP", "DATETIME")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _has_temporal_column(table: dict) -> bool:
    """Return True if the table has at least one date/time column.

    A column is considered temporal if its type name contains a date/time
    keyword OR its min value looks like an ISO date string (YYYY-MM-DD).
    The ISO-value check is needed for SQLite, which stores dates as TEXT.
    """
    for col in table.get("columns", []):
        col_type = col.get("type", "").upper()
        mn = col.get("stats", {}).get("min")
        if any(h in col_type for h in _DATE_TYPE_HINTS):
            return True
        if isinstance(mn, str) and _ISO_DATE_RE.match(mn):
            return True
    return False


def _deterministic_catalogue_fields(snapshot: dict) -> dict:
    """Compute catalogue fields that are fully determined by the schema snapshot.

    Returns a dict with:
      table_relationships  — from FK declarations in the schema
      time_coverage        — MIN/MAX across all date/timestamp columns
      tables_meta          — {table_name: {row_count, data_quality_notes}}
      data_quality_notes   — database-wide null-rate and small-table notes
    """
    # ── table_relationships ────────────────────────────────────────────────────
    relationships: list[dict] = []
    for tbl in snapshot["tables"]:
        for fk in tbl.get("foreign_keys", []):
            from_cols = fk.get("from_cols", [])
            to_cols   = fk.get("to_cols", [])
            if len(from_cols) == 1 and len(to_cols) == 1:
                relationships.append({
                    "table_a":  tbl["name"],
                    "table_b":  fk["to_table"],
                    "join_key": from_cols[0],
                })

    # ── time_coverage ─────────────────────────────────────────────────────────
    all_mins: list[str] = []
    all_maxs: list[str] = []
    for tbl in snapshot["tables"]:
        for col in tbl.get("columns", []):
            col_type = col.get("type", "").upper()
            stats = col.get("stats", {})
            mn, mx = stats.get("min"), stats.get("max")
            type_is_date = any(h in col_type for h in _DATE_TYPE_HINTS)
            value_looks_like_date = isinstance(mn, str) and bool(_ISO_DATE_RE.match(mn))
            if not (type_is_date or value_looks_like_date):
                continue
            if mn:
                all_mins.append(str(mn)[:10])
            if mx:
                all_maxs.append(str(mx)[:10])
    time_coverage = {
        "start": min(all_mins) if all_mins else "",
        "end":   max(all_maxs) if all_maxs else "",
    }

    # ── per-table null-rate notes + row_count ─────────────────────────────────
    tables_meta: dict[str, dict] = {}
    db_wide_notes: list[str] = []
    for tbl in snapshot["tables"]:
        row_count = tbl["row_count"]
        notes: list[str] = []
        for col in tbl.get("columns", []):
            n_null = col.get("stats", {}).get("n_null", 0)
            if row_count > 0 and n_null > 0:
                rate = n_null / row_count
                if rate >= _NULL_RATE_THRESHOLD:
                    notes.append(
                        f"`{col['name']}` has {n_null:,} nulls "
                        f"({rate * 100:.0f}% of rows)"
                    )
        tables_meta[tbl["name"]] = {"row_count": row_count, "data_quality_notes": notes}
        if 0 < row_count < _SMALL_TABLE_ROWS:
            db_wide_notes.append(
                f"`{tbl['name']}` is a small table ({row_count} rows) — "
                "metrics built on it may cover only a narrow slice of data."
            )

    return {
        "table_relationships": relationships,
        "time_coverage":       time_coverage,
        "tables_meta":         tables_meta,
        "data_quality_notes":  db_wide_notes,
    }


def _build_catalogue(data: dict, snapshot: dict) -> DataCatalogue:
    det = _deterministic_catalogue_fields(snapshot)

    # Merge per-table deterministic fields (row_count, data_quality_notes) with
    # the LLM's per-table descriptions and key_columns.
    tables_out = []
    for tbl in data.get("tables", []):
        name = tbl.get("name", "") if isinstance(tbl, dict) else ""
        meta = det["tables_meta"].get(name, {})
        tables_out.append({
            "name":               name,
            "row_count":          meta.get("row_count", tbl.get("row_count", 0)),
            "description":        tbl.get("description", "") if isinstance(tbl, dict) else "",
            "key_columns":        tbl.get("key_columns", []) if isinstance(tbl, dict) else [],
            "data_quality_notes": meta.get("data_quality_notes", []),
        })

    # LLM may provide cross-table semantic notes; prepend the deterministic ones.
    llm_dq = data.get("data_quality_notes") or []
    if isinstance(llm_dq, str):
        llm_dq = [llm_dq]
    combined_dq = det["data_quality_notes"] + [n for n in llm_dq if isinstance(n, str)]

    try:
        return DataCatalogue(
            analysed_at=datetime.now().isoformat(timespec="seconds"),
            model=_config.model,
            connection=snapshot["connection_string"],
            dialect=snapshot["dialect"],
            description=data.get("description") or "",
            overview=data.get("overview") or "",
            tables=tables_out,
            measurable_metrics=data["measurable_metrics"],
            queryable_facts=data.get("queryable_facts") or [],
            time_coverage=det["time_coverage"],
            data_quality_notes=combined_dq,
            key_terms=data.get("key_terms") or [],
            table_relationships=det["table_relationships"],
        )
    except Exception as exc:
        console.print(f"[red]Catalogue validation error: {exc}[/red]")
        console.print(f"[dim]Keys received: {list(data.keys())}[/dim]")
        raise


def _decode_sql_error(exc: Exception) -> str:
    """Extract a concise, readable root cause from a SQLAlchemy exception."""
    msg = str(exc)
    # SQLAlchemy wraps the DB error — strip the ORM preamble to surface the actual cause
    for marker in (
        "(sqlite3.OperationalError)",
        "(psycopg2.",
        "(pymysql.err.",
        "sqlalchemy.exc.",
    ):
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

    n_tables = len(catalogue.tables)
    n_metrics = len(catalogue.measurable_metrics)
    n_facts = len(catalogue.queryable_facts)
    fact_word = "fact" if n_facts == 1 else "facts"
    date = catalogue.analysed_at[:10]
    tr_start = catalogue.time_coverage.start[:7]
    tr_end = catalogue.time_coverage.end[:7]
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
            lines.append(f'    {a} -->|"<u><i><b>{rel.join_key}</b></i></u>"| {b}')
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
    """Write catalogue JSON to an auto-indexed path, claimed atomically.

    out_path is a pattern without index or extension, e.g.:
        ../DBs/gemini-2.5-flash/saas_catalogue

    The final filename is  <pattern>_<n>.json  where <n> is the lowest
    integer not already on disk.  O_EXCL guarantees that two concurrent
    processes never write to the same file even if they computed the same
    index at startup.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    idx = 1
    while True:
        candidate = p.parent / f"{p.name}_{idx}.json"
        try:
            fd = os.open(str(candidate), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(fd, "w") as f:
                json.dump(catalogue.model_dump(), f, indent=2)
            break
        except FileExistsError:
            idx += 1

    console.print(f"\n[bold green]Catalogue written to {candidate}[/bold green]")

    if catalogue.overview:
        overview_path = candidate.parent / candidate.name.replace("_catalogue_", "_overview_").replace(
            ".json", ".md"
        )
        overview_path.write_text(_render_overview_md(catalogue))
        console.print(f"[bold green]Overview written to {overview_path}[/bold green]")


# ── phase 3 — refinement ───────────────────────────────────────────────────────

_DIRECTLY_PATCHED = {"date_mismatch", "period_boundary"}


def _filter_agent_issues(
    warn_metrics: list[tuple],
    warn_facts: list[tuple],
) -> list[tuple]:
    """Return only the (item, result) pairs that need agent investigation.

    Excluded:
      - Metrics whose only issues are date_mismatch / period_boundary (patched directly).
      - Any result where the SQL ran fine but the eval framework crashed
        (sql_ok=True, error starts with "eval error:") — the agent cannot fix
        an environment problem like a missing numpy module.
    """
    return [
        (m, r)
        for m, r in warn_metrics
        if set((r.error or "").split(", ")) - _DIRECTLY_PATCHED
        and not _is_evaluator_crash(r)
    ] + [(f, r) for f, r in warn_facts if not _is_evaluator_crash(r)]


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
    console.print(
        Panel(
            "[bold cyan]Phase 3 — Refinement[/bold cyan]\n"
            "[dim]Evaluating catalogue against live database…[/dim]",
            border_style="cyan",
            padding=(0, 1),
        )
    )

    # ── run eval ───────────────────────────────────────────────────────────────
    # Each SQL is run against the full database (no row limit) so row counts,
    # null rates, and date ranges are exact — not sampled.
    console.print(
        f"[dim]  Running {len(catalogue.measurable_metrics)} metric(s) and "
        f"{len(catalogue.queryable_facts)} fact(s) against the full database…[/dim]"
    )
    metric_results = [
        evaluate_metric(engine, m.model_dump()) for m in catalogue.measurable_metrics
    ]
    fact_results = [
        evaluate_fact(engine, f.model_dump()) for f in catalogue.queryable_facts
    ]

    warn_metrics = [
        (catalogue.measurable_metrics[i], metric_results[i])
        for i, r in enumerate(metric_results)
        if r.status in ("WARN", "FAIL")
    ]
    warn_facts = [
        (catalogue.queryable_facts[i], fact_results[i])
        for i, r in enumerate(fact_results)
        if r.status in ("WARN", "FAIL")
    ]

    # ── check table coverage ───────────────────────────────────────────────────
    db_tables = set(sqla_inspect(engine).get_table_names())
    mentioned = {t for m in catalogue.measurable_metrics for t in m.tables_used}
    mentioned |= {t for f in catalogue.queryable_facts for t in f.tables_used}
    mentioned |= {t.name for t in catalogue.tables}
    uncovered = sorted(db_tables - mentioned)
    if uncovered:
        console.print(
            f"  [yellow]⚠ Tables with no metrics or facts:[/yellow] {', '.join(uncovered)}"
        )

    if not warn_metrics and not warn_facts:
        console.print(
            "[green]  No metric/fact issues found — skipping refinement.[/green]\n"
        )
        return catalogue, metric_results, fact_results, uncovered

    # ── display issues ─────────────────────────────────────────────────────────
    console.print(
        f"  [bold]Issues found ({len(warn_metrics) + len(warn_facts)}):[/bold]"
    )
    for m, r in warn_metrics:
        console.print(f"    [yellow]⚠ {r.status}[/yellow]  {m.name}  —  {r.error}")
    for f, r in warn_facts:
        console.print(
            f"    [yellow]⚠ {r.status}[/yellow]  (fact) {f.name}  —  {r.error}"
        )

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
    patched_metrics = {m.name: m for m in catalogue.measurable_metrics}
    date_patched: list[str] = []

    for m, r in warn_metrics:
        codes = set((r.error or "").split(", "))

        if "date_mismatch" in codes and r.actual_start and r.actual_end:
            new_start = r.actual_start
            new_end = r.actual_end
        else:
            new_start = m.time_range.start
            new_end = m.time_range.end

        if "period_boundary" in codes:
            new_start = _snap_to_period(new_start, m.granularity)
            new_end = _snap_to_period(new_end, m.granularity)

        if new_start != m.time_range.start or new_end != m.time_range.end:
            patched_metrics[m.name] = m.model_copy(
                update={"time_range": TimeRange(start=new_start, end=new_end)}
            )
            date_patched.append(
                f"    [green]✓[/green] {m.name}: "
                f"[dim]{m.time_range.start[:7]} → {m.time_range.end[:7]}[/dim] → "
                f"[green]{new_start[:7]} → {new_end[:7]}[/green]"
                + (
                    f" [dim](period_boundary)[/dim]"
                    if "period_boundary" in codes
                    else ""
                )
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
        console.print(
            "  [green]All issues patched directly — no agent call needed.[/green]\n"
        )
        final = catalogue.model_copy(
            update={"measurable_metrics": list(patched_metrics.values())}
        )
        final_mr = [
            evaluate_metric(engine, m.model_dump()) for m in final.measurable_metrics
        ]
        final_fr = [
            evaluate_fact(engine, f.model_dump()) for f in final.queryable_facts
        ]
        return final, final_mr, final_fr, uncovered

    # ── build refinement prompt ────────────────────────────────────────────────
    issues_text = ""
    for item, r in agent_issues:
        is_metric = hasattr(item, "time_range")
        kind = "METRIC" if is_metric else "FACT"
        # Append evaluator measurements so the model can act without a confirmation query.
        # For zero_rows it can immediately decide to remove; for constant_values it can
        # see the value and row count and write notes directly.
        result_detail = ""
        v_min = getattr(r, "value_min", None)
        v_max = getattr(r, "value_max", None)
        if r.n_rows == 0:
            result_detail = "  Eval result: 0 rows returned\n"
        elif v_min is not None and v_min == v_max:
            result_detail = (
                f"  Eval result: {r.n_rows} row(s), all values = {v_min}"
                f" (genuinely sparse — check if the source table is small)\n"
            )
        elif r.n_rows > 0:
            result_detail = f"  Eval result: {r.n_rows} row(s) returned\n"
        issues_text += (
            f"\n{kind}: {item.name}\n"
            f"  Issue: {r.error}\n"
            f"{result_detail}"
            f"  SQL: {item.sql}\n"
        )

    current_catalogue_json = _compact_json(
        catalogue.model_copy(
            update={"measurable_metrics": list(patched_metrics.values())}
        ).model_dump()
    )

    refinement_prompt = (
        f"Here is the schema snapshot:\n\n```\n{schema_text}\n```\n\n"
        f"Here is the current catalogue:\n\n```json\n{current_catalogue_json}\n```\n\n"
        f"The following issues were found by the evaluator:\n{issues_text}\n"
        f"Investigate each issue using run_query, fix what you can, "
        f"then submit the complete corrected catalogue via finish_catalogue."
    )

    console.print(
        Panel(
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
        )
    )

    # ── run refinement loop ────────────────────────────────────────────────────
    # Mark the refinement prompt for caching on the Anthropic path — it contains
    # the full schema + catalogue JSON and is repeated across all refinement iterations.
    backend = _make_backend(refinement_prompt, REFINEMENT_SYSTEM_PROMPT)

    refined_data, _, _, _ = _run_phase(
        backend,
        engine,
        tools=[_RUN_QUERY_TOOL, _FINISH_CATALOGUE_TOOL],
        max_iter=_REFINEMENT_BUDGET,
        phase_label="3 (refinement)",
        usage=usage,
        table_columns=table_columns,
        tracker=tracker,
        initial_max_tokens=_phase2_max_tokens(),
        nudge_text=(
            "Your previous response was empty — the API returned no content. "
            "Please continue: use run_query to investigate the reported issues, "
            "then call finish_catalogue with the complete corrected catalogue."
        ),
    )

    if refined_data is None:
        console.print(
            "[yellow]  Refinement agent did not produce a valid catalogue — "
            "keeping directly-patched catalogue.[/yellow]"
        )
        final = catalogue.model_copy(
            update={"measurable_metrics": list(patched_metrics.values())}
        )
        final_mr = [
            evaluate_metric(engine, m.model_dump()) for m in final.measurable_metrics
        ]
        final_fr = [
            evaluate_fact(engine, f.model_dump()) for f in final.queryable_facts
        ]
        return final, final_mr, final_fr, uncovered

    refined_catalogue = _build_catalogue(
        refined_data,
        {
            "connection_string": catalogue.connection,
            "dialect": catalogue.dialect,
            "tables": [],
        },
    )
    # Restore deterministic fields from the Phase 1 catalogue — schema facts
    # (FK relationships, time coverage, row counts, null-rate notes) don't
    # change during refinement, but the stub snapshot above produces blanks.
    orig_tables_by_name = {t.name: t for t in catalogue.tables}
    refined_catalogue = refined_catalogue.model_copy(update={
        "table_relationships": catalogue.table_relationships,
        "time_coverage":       catalogue.time_coverage,
        "tables": [
            t.model_copy(update={
                "row_count":          orig_tables_by_name[t.name].row_count,
                "data_quality_notes": orig_tables_by_name[t.name].data_quality_notes,
            }) if t.name in orig_tables_by_name else t
            for t in refined_catalogue.tables
        ],
    })
    refined_catalogue = _drop_broken_sql(refined_catalogue, engine)

    patched_catalogue = catalogue.model_copy(
        update={"measurable_metrics": list(patched_metrics.values())}
    )
    refined_catalogue, did_fallback = _select_phase3_result(
        refined_catalogue, patched_catalogue
    )

    if did_fallback:
        console.print(
            "[yellow]  ⚠ Refinement produced an empty catalogue after SQL validation "
            "(all metrics had broken SQL) — falling back to the directly-patched catalogue.[/yellow]"
        )
        final_mr = [
            evaluate_metric(engine, m.model_dump())
            for m in refined_catalogue.measurable_metrics
        ]
        final_fr = [
            evaluate_fact(engine, f.model_dump())
            for f in refined_catalogue.queryable_facts
        ]
        return refined_catalogue, final_mr, final_fr, uncovered

    n_orig = len(catalogue.measurable_metrics)
    n_refined = len(refined_catalogue.measurable_metrics)
    if n_orig > 0 and n_refined < n_orig * 0.5:
        console.print(
            f"[yellow]  ⚠ Refinement dropped {n_orig - n_refined} of {n_orig} metrics "
            f"({n_refined} remain). Consider re-running with a larger refinement budget.[/yellow]"
        )

    # ── phase 3 summary ───────────────────────────────────────────────────────
    orig_names = {m.name for m in catalogue.measurable_metrics}
    refined_names = {m.name for m in refined_catalogue.measurable_metrics}
    removed = orig_names - refined_names
    added = refined_names - orig_names

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

    console.print(
        Panel(
            "[bold green]Phase 3 complete[/bold green]\n" + "\n".join(lines),
            border_style="green",
            padding=(0, 1),
        )
    )

    final_mr = [
        evaluate_metric(engine, m.model_dump())
        for m in refined_catalogue.measurable_metrics
    ]
    final_fr = [
        evaluate_fact(engine, f.model_dump()) for f in refined_catalogue.queryable_facts
    ]
    return refined_catalogue, final_mr, final_fr, uncovered


# ── Anthropic client ───────────────────────────────────────────────────────────


def _get_anthropic_client():
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Add it to .env or tshe environment."
        )
    return anthropic.Anthropic(api_key=api_key)
