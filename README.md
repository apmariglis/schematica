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

---

## Installation

```bash
# requires uv
uv sync

# with LiteLLM support (for Gemini and other providers)
uv sync --extra litellm
```

Copy `.env.example` to `.env` and fill in your API key and model choice.

---

## Usage

```bash
# Analyse a database — output goes to <db_dir>/<model>/<db>_catalogue_<n>.json
uv run schematica --db sqlite:///path/to/mydb.db

# Specify a custom output path
uv run schematica --db sqlite:///path/to/mydb.db --out path/to/catalogue.json

# Evaluate an existing catalogue
uv run python scripts/eval_catalogue.py --catalogue path/to/mydb_catalogue_1.json

# Compare catalogues produced by different models for the same database
uv run python scripts/compare_catalogues.py --data path/to/catalogues/
uv run python scripts/compare_catalogues.py --data path/to/catalogues/ --db mydb
uv run python scripts/compare_catalogues.py --data path/to/catalogues/ --judge   # adds LLM semantic scoring
```

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
