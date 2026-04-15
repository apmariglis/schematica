# schematica

Agentic LLM explorer that catalogues every metric and queryable fact in a SQL database.

Point schematica at any SQL database and it will autonomously explore the schema, validate what it finds, and produce a `data_catalogue.json` — a structured map of every time-series metric and queryable fact the database can serve.

---

## How it works

Schematica runs in three phases:

**Phase 1 — Exploration**
The agent receives a schema snapshot (table structure, column types, statistics, sample rows — no LLM cost for this part) and explores the database by running SQL queries. It discovers join paths, infers date columns, and proposes named metrics that each return a `(date, value)` series.

**Phase 2 — Documentation**
The agent compiles everything it learned into a structured catalogue with descriptions, time ranges, granularity, units, and confidence ratings.

**Phase 3 — Validation**
Every metric and fact is executed against the full database. Issues like wrong shape, zero rows, high nulls, date mismatches, duplicate SQL, and constant values are detected. Many are auto-patched; the rest are sent back to the agent for correction or removed with a note.

Schematica is **read-only** — it never modifies your database. For SQLite this is enforced at the driver level (`mode=ro`). For PostgreSQL, MySQL, and other databases **you should connect with a dedicated user that has only `SELECT` privileges** — that is the only reliable enforcement mechanism for those dialects, and strongly recommended before pointing schematica at any production database.

---

## Getting started

**1. Install**

```bash
uv sync                        # Anthropic only
uv sync --extra litellm        # add Gemini and other providers
```

**2. Configure**

Copy `.env.example` to `.env` and set the API key for your provider:

| Provider | Key | Model prefix |
|---|---|---|
| Gemini | `GOOGLE_API_KEY` | `gemini/gemini-2.5-flash` |
| Anthropic (Sonnet) | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-4-6` |
| Anthropic (Opus) | `ANTHROPIC_API_KEY` | `anthropic/claude-opus-4-6` |

Set `SC_MODEL` in `.env` to your chosen model. Gemini is the default — it performs well and is cheap to run.

**3. Run against a database**

Pass a file path or a full SQLAlchemy connection string — both work:

```bash
uv run schematica --db path/to/mydb.db                     # SQLite file path (auto-converted)
uv run schematica --db sqlite:///path/to/mydb.db           # explicit SQLite connection string
uv run schematica --db postgresql://user:pw@host/mydb      # PostgreSQL
uv run schematica --db mysql://user:pw@host/mydb           # MySQL
uv run schematica --db mssql+pyodbc://user:pw@dsn          # SQL Server
uv run schematica --db oracle+cx_oracle://user:pw@host/sid # Oracle
```

Any database supported by SQLAlchemy works. Output is written to `<db_dir>/<model>/<db_stem>_catalogue_<n>.json`. For example, running against `data/sales.db` with `gemini-2.5-flash` produces `data/gemini-2.5-flash/sales_catalogue_1.json`. Each run gets an auto-incremented index so repeated runs never overwrite each other.

Use `--model` to override `SC_MODEL` from `.env` for a single run — useful for comparing models without editing config:

```bash
uv run schematica --db path/to/mydb.db --model gpt-4o
uv run schematica --db path/to/mydb.db --model together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo
```

**4. Evaluate the catalogue**

```bash
uv run python scripts/eval_catalogue.py \
    --db path/to/mydb.db \
    --catalogue path/to/mydb_catalogue_1.json
```

Runs every metric and queryable fact against the live database and produces a quality report (pass/warn/fail per metric, null rates, date range accuracy, table coverage).

**5. Compare models**

Run schematica multiple times with different `SC_MODEL` values, then compare the catalogues they produced:

```bash
# scan a data directory for all matching catalogues
uv run python scripts/compare_catalogues.py \
    --dbs path/to/mydb.db \
    --catalogues data/

# multiple databases
uv run python scripts/compare_catalogues.py \
    --dbs path/to/db1.db postgresql://user:pw@host/db2 \
    --catalogues data/

# catalogues scattered across different folders
uv run python scripts/compare_catalogues.py \
    --dbs path/to/mydb.db \
    --catalogues runs/2025-01/mydb_catalogue_1.json \
                 runs/2025-02/mydb_catalogue_1.json \
                 team/alice/mydb_catalogue_1.json

# add LLM semantic scoring of metric descriptions
uv run python scripts/compare_catalogues.py \
    --dbs path/to/mydb.db \
    --catalogues data/ \
    --judge
```

`--dbs` accepts file paths (`.db`, `.sqlite`), SQLAlchemy connection strings, or a directory (expanded to all `.db` and `.sqlite` files inside it). `--catalogues` accepts a directory (catalogues found directly inside it and in its immediate subfolders are included) or an explicit list of JSON files. Each catalogue is matched to its database via the `connection` field stored inside the catalogue JSON — so mixing databases and scattered files all works. The model label is taken from the catalogue file's parent directory name.

---

## Progress display

While running, schematica prints a live stats box after each iteration:

```
  Phase 1 (exploration) — iteration 4/31…
  [tool calls, query results…]
  ╭─ current iter 4/31 ─────────────────────────────────────────────╮
  │  7,104 in · 201 out · $0.0026 · 1.7s · 3.6% ctx                │
  ├─ accumulated ───────────────────────────────────────────────────┤
  │  28,419 in · 804 out · $0.0104 · 18.5s · 13.0 iter/min         │
  ╰─ each iteration = 1 LLM call ──────────────────────────────────╯
  Phase 1 (exploration) — iteration 5/31…
```

**current iter** — tokens sent/received, cost, and wall-clock time for that single LLM call, plus how full the context window is (`% ctx`).

**accumulated** — session totals: tokens, cost, elapsed time, and average throughput in iterations per minute.

Context window fill (`% ctx`) is shown when the model is recognised. It is derived from the input token count, which equals the full conversation history sent on each call.

---

## Output

`data_catalogue.json` contains:

```json
{
  "description": "One-sentence summary of the database domain",
  "tables": [...],
  "measurable_metrics": [
    {
      "name": "monthly_total_revenue",
      "description": "Total revenue per month",
      "sql": "SELECT DATE_TRUNC(created_at, MONTH), SUM(amount) FROM orders GROUP BY 1",
      "time_range": {"start": "2022-01-01", "end": "2024-12-31"},
      "granularity": "monthly",   // daily | weekly | monthly | quarterly | annual | tick (un-aggregated, one row per raw event)
      "unit": "€",
      "tables_used": ["orders"],
      "confidence": "high",
      "agent_notes": "..."
    }
  ],
  "queryable_facts": [
    {
      "name": "region_lookup",
      "description": "Mapping of region codes to names",
      "sql": "SELECT DISTINCT region_code, region_name FROM regions",
      "tables_used": ["regions"],
      "agent_notes": "Static reference table"
    }
  ],
  "time_coverage": {"start": "2022-01-01", "end": "2024-12-31"},
  "data_quality_notes": [...]
}
```

---

## Configuration

All settings are prefixed `SC_` in your `.env` file:

| Variable | Default | Description |
|---|---|---|
| `SC_MODEL` | `gemini/gemini-2.5-flash` | LLM model — use `gemini/` prefix for Gemini via LiteLLM; use `anthropic/` prefix to enable native Anthropic SDK with prompt caching |
| `SC_CACHE` | `false` | Prompt caching (Anthropic native SDK only) |
| `SC_MAX_ROWS` | `5` | Max rows returned per query during exploration |
| `SC_MAX_CHARS` | `500` | Max characters per query result |
| `SC_BUDGET_BASE` | `10` | Min exploration iterations |
| `SC_BUDGET_MULTIPLIER` | `3` | Extra iterations per table |
| `SC_BUDGET_CAP` | `50` | Max exploration iterations |
| `SC_REFINEMENT_BUDGET` | `15` | Max Phase 3 refinement iterations |
| `SC_MAX_OUTPUT_TOKENS` | `32768` | Max tokens per LLM call |

---

## Exploration budget

The Phase 1 iteration budget scales with the number of tables in the database:

```
budget = min(SC_BUDGET_BASE + n_tables × SC_BUDGET_MULTIPLIER, SC_BUDGET_CAP)
```

With the defaults (`base=10`, `multiplier=3`, `cap=50`):

| Tables | Budget |
|--------|--------|
| 1 | 13 |
| 5 | 25 |
| 10 | 40 |
| 14+ | 50 (capped) |

The agent must use at least half the budget before finishing Phase 1, so small databases still get a minimum of 5 exploratory queries.

---

## Phase 3 validation codes

After Phase 2 the catalogue is validated against the live database. Issues are reported with short codes:

| Code | Meaning | Action |
|------|---------|--------|
| `zero_rows` | SQL ran without error but returned 0 rows — filter condition may be wrong or data is absent | Sent to refinement agent |
| `sparse` | Fewer than 3 rows returned — not enough data points for a reliable metric | Sent to refinement agent |
| `high_nulls` | Value column has >10% NULL entries — may silently skew aggregations | Sent to refinement agent |
| `extra_cols` | Query returns more than 2 columns — metrics must return exactly date + value | Sent to refinement agent |
| `constant_values` | All non-null rows have the same value — the metric carries no trend information | Sent to refinement agent |
| `non_date_col` | First column (date column) cannot be parsed as dates in >5% of rows | Sent to refinement agent |
| `date_mismatch` | Actual data range falls outside the declared `time_range` | Auto-patched |
| `period_boundary` | `time_range` start/end does not align to the granularity boundary (e.g. monthly → first of month) | Auto-patched |

Auto-patched issues are corrected directly without an LLM call. Everything else is fed back to the refinement agent (Phase 3), which uses `run_query` to investigate and resubmits a corrected catalogue.

---

## Troubleshooting

**`litellm` not found**
Run `uv sync --extra litellm`. The base install only includes the Anthropic SDK.

**`API key not set` / `401 Unauthorized`**
Check that the correct key is set in `.env` for your provider (see the table in Getting started, step 2).

**`OperationalError: unable to open database file`**
The path passed to `--db` / `--dbs` does not exist. Check the path and working directory.

**`OperationalError: attempt to write a readonly database`**
Schematica opens SQLite in read-only mode (`mode=ro`). If you see this for a non-SQLite database, the connected user has write access — connect with a read-only user.

**`UserWarning: Fuzzy match: '...' resolved to '...'`**
`broker.fetch()` applies fuzzy name matching. The warning means the name you passed was close but not exact — use the resolved name shown in the warning (or the exact name from `broker.list_metrics()`) to suppress it.

**Empty or very sparse catalogue**
The database may have no date/time columns. Schematica can only produce time-series metrics when a date dimension is present. Queryable facts (static lookup tables, snapshots) will still be catalogued.

**Phase 3 refinement loop hits budget**
Increase `SC_REFINEMENT_BUDGET` in `.env`. The default is 15 iterations. Complex schemas with many SQL issues may need more.

**`No output file written` — agent exhausted retries**
The agent hit the output token limit while writing `finish_catalogue`, which truncates the JSON mid-response. The agent is asked to resubmit a smaller catalogue (target 3–5 metrics per table, ~70 total). If this keeps failing, increase `SC_MAX_OUTPUT_TOKENS` in `.env` (default 32768) or reduce the exploration budget so the agent proposes fewer metrics.

---

## Supported databases

Any database supported by SQLAlchemy: SQLite, PostgreSQL, MySQL, MSSQL, and others.
