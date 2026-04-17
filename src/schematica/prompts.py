"""
prompts.py — System prompts, tool definitions, and validation constants.

Pure data module: no runtime state, no side effects.
"""
from __future__ import annotations


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
- constant_values: Every period returns the same value (value_min == value_max). \
  Run a query to verify the SQL is correct. If the SQL is correct and the source \
  table is genuinely small or sparse (e.g. only a handful of events total), the \
  constant output is real data — keep the metric and record an explanation in \
  agent_notes (e.g. "Only 18 rows in source table; all months show 1-2 events"). \
  Only remove or rewrite the metric if there is a clear SQL logic error — for \
  example a missing GROUP BY that causes every row to repeat the same aggregate.

Submit the COMPLETE corrected catalogue via finish_catalogue — include all \
metrics (fixed and unchanged), not just the ones you fixed. Remove only metrics \
that genuinely have no data.
"""


def make_run_query_tool(max_rows: int) -> dict:
    """Build the run_query tool definition for the given row limit."""
    return {
        "name": "run_query",
        "description": (
            f"Execute a read-only SQL SELECT query. Results truncated to {max_rows} rows. "
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
