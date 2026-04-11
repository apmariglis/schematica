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

Schematica is **read-only** — it never modifies your database. All connections are opened in read-only mode and only `SELECT` queries are permitted.

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
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-4-20250514` |

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
uv run python scripts/compare_catalogues.py \
    --db path/to/sales.db \
    --data data/

uv run python scripts/compare_catalogues.py \
    --db path/to/sales.db \
    --data data/ \
    --judge   # adds LLM semantic scoring of metric descriptions
```

`--db` determines which catalogues to compare: the stem is extracted (e.g. `sales` from `sales.db`) and the script finds all files matching `data/*/sales_catalogue_*.json`. Given a layout like:

```
data/
  gemini-2.5-flash/sales_catalogue_1.json
  claude-sonnet-4/sales_catalogue_1.json
```

…it loads one catalogue per model and produces a side-by-side comparison.

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
      "granularity": "monthly",
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

## Supported databases

Any database supported by SQLAlchemy: SQLite, PostgreSQL, MySQL, MSSQL, and others.
