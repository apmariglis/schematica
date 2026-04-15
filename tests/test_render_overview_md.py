"""
Tests for _render_overview_md — v6 structured Markdown overview of a catalogue.

Structure:
  # {description}   ← short title-worthy name, e.g. "SaaS Business Database"
  > {date} · {model} · N tables · N metrics · N facts · YYYY-MM → YYYY-MM

  ## Overview
  {narrative}

  ## Key Terms          ← only when key_terms exist
  - **Term** — definition

  ## Data Quality Notes  ← only when catalogue-level notes exist
  1. ...

  ## Tables at a Glance
  | Table | Rows | What it holds |

  ## Table Relationships  ← only when table_relationships exist
  ```mermaid
  ...
  ```
  > Legend: ...

  ## Tables Reference
  ### <u>**table_name**</u>
  description
  Columns: <u>***col1***</u>, ...
  > **Data notes**
  > - ...

  ## Metrics
  > thematic grouping note
  ### {group}
  - **Metric Name**
    ***Frequency:*** ... · ***Unit:*** ... · ***Range:*** ...
    description
    Tables: ...
    ```sql
    ...
    ```

  ## Facts
  > explanation note
  ### Fact Name
  ...
"""
from __future__ import annotations

from schematica.agent import _render_overview_md
from schematica.catalogue import (
    DataCatalogue, KeyTerm, MeasurableMetric, QueryableFact,
    TableRelationship, TableSummary, TimeRange,
)


# ── helpers ─────────────────────────────────────────────────────────────────

def _metric(name: str, tables: list[str], granularity: str = "monthly",
            description: str = "A metric.", sql: str = "SELECT dt, v FROM t",
            unit: str = "count", confidence: str = "high",
            start: str = "2023-01-01", end: str = "2023-12-01",
            agent_notes: str = "", group: str = "") -> MeasurableMetric:
    return MeasurableMetric(
        name=name, description=description, sql=sql,
        time_range=TimeRange(start=start, end=end),
        granularity=granularity, unit=unit,
        tables_used=tables, confidence=confidence,
        agent_notes=agent_notes, group=group,
    )


def _fact(name: str, tables: list[str],
          description: str = "A fact.", sql: str = "SELECT * FROM t",
          agent_notes: str = "") -> QueryableFact:
    return QueryableFact(
        name=name, description=description, sql=sql,
        tables_used=tables, agent_notes=agent_notes,
    )


def _table(name: str, rows: int = 100,
           description: str = "A table.",
           key_columns: list[str] | None = None,
           data_quality_notes: list[str] | None = None) -> TableSummary:
    return TableSummary(
        name=name, row_count=rows, description=description,
        key_columns=key_columns or ["id", "created_at"],
        data_quality_notes=data_quality_notes or [],
    )


def _catalogue(
    tables: list[TableSummary] | None = None,
    metrics: list[MeasurableMetric] | None = None,
    facts: list[QueryableFact] | None = None,
    description: str = "Solar Wind Energy Database",
    overview: str = "This database tracks solar and wind energy assets.",
    dq_notes: list[str] | None = None,
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    model: str = "claude-haiku-4-5",
    analysed_at: str = "2025-04-14T10:00:00",
    key_terms: list[KeyTerm] | None = None,
    table_relationships: list[TableRelationship] | None = None,
) -> DataCatalogue:
    return DataCatalogue(
        connection="sqlite:///test.db",
        dialect="sqlite",
        description=description,
        overview=overview,
        model=model,
        analysed_at=analysed_at,
        tables=tables or [_table("t")],
        measurable_metrics=metrics or [],
        queryable_facts=facts or [],
        time_coverage=TimeRange(start=start, end=end),
        data_quality_notes=dq_notes or [],
        key_terms=key_terms or [],
        table_relationships=table_relationships or [],
    )


# ── document header ──────────────────────────────────────────────────────────

def test_title_is_the_description_field():
    md = _render_overview_md(_catalogue(description="Solar Wind DB"))
    assert "# Solar Wind DB" in md


def test_title_fallback_when_description_is_empty():
    md = _render_overview_md(_catalogue(description=""))
    assert "# Database Overview" in md


def test_metadata_line_contains_analysed_date():
    md = _render_overview_md(_catalogue(analysed_at="2025-04-14T10:00:00"))
    assert "2025-04-14" in md


def test_metadata_line_contains_model():
    md = _render_overview_md(_catalogue(model="claude-sonnet-4-6"))
    assert "claude-sonnet-4-6" in md


def test_metadata_line_contains_table_count():
    md = _render_overview_md(_catalogue(tables=[_table("a"), _table("b"), _table("c")]))
    assert "3 tables" in md


def test_metadata_line_contains_metric_count():
    md = _render_overview_md(_catalogue(metrics=[_metric("m1", ["t"]), _metric("m2", ["t"])]))
    assert "2 metrics" in md


def test_metadata_line_contains_fact_count():
    md = _render_overview_md(_catalogue(facts=[_fact("f1", ["t"])]))
    assert "1 fact" in md


def test_metadata_line_contains_time_range():
    md = _render_overview_md(_catalogue(start="2022-01-01", end="2024-12-31"))
    assert "2022-01" in md
    assert "2024-12" in md


# ── overview section ─────────────────────────────────────────────────────────

def test_overview_section_heading_present():
    md = _render_overview_md(_catalogue())
    assert "## Overview" in md


def test_overview_narrative_present():
    md = _render_overview_md(_catalogue(overview="Covers renewable energy assets globally."))
    assert "Covers renewable energy assets globally." in md


# ── key terms ────────────────────────────────────────────────────────────────

def test_key_terms_section_present_when_terms_exist():
    md = _render_overview_md(_catalogue(
        key_terms=[KeyTerm(term="MRR", definition="Monthly Recurring Revenue.")]
    ))
    assert "## Key Terms" in md


def test_key_terms_section_absent_when_no_terms():
    md = _render_overview_md(_catalogue(key_terms=[]))
    assert "## Key Terms" not in md


def test_key_term_name_and_definition_in_output():
    md = _render_overview_md(_catalogue(
        key_terms=[KeyTerm(term="Churn", definition="When a customer cancels.")]
    ))
    assert "**Churn**" in md
    assert "When a customer cancels." in md


# ── catalogue-level data quality notes ───────────────────────────────────────

def test_data_quality_section_present_when_notes_exist():
    md = _render_overview_md(_catalogue(dq_notes=["Note one.", "Note two."]))
    assert "## Data Quality Notes" in md


def test_data_quality_notes_listed():
    md = _render_overview_md(_catalogue(dq_notes=["Gap in 2023-Q2.", "Nulls in cost column."]))
    assert "Gap in 2023-Q2." in md
    assert "Nulls in cost column." in md


def test_data_quality_section_absent_when_no_notes():
    md = _render_overview_md(_catalogue(dq_notes=[]))
    assert "## Data Quality Notes" not in md


# ── tables at a glance ───────────────────────────────────────────────────────

def test_tables_at_a_glance_section_present():
    md = _render_overview_md(_catalogue())
    assert "## Tables at a Glance" in md


def test_tables_at_a_glance_contains_all_table_names():
    md = _render_overview_md(_catalogue(tables=[_table("orders"), _table("products")]))
    assert "orders" in md
    assert "products" in md


def test_tables_at_a_glance_contains_row_counts():
    md = _render_overview_md(_catalogue(tables=[_table("orders", rows=1234)]))
    assert "1,234" in md


# ── table relationships ───────────────────────────────────────────────────────

def test_table_relationships_section_present_when_relationships_exist():
    md = _render_overview_md(_catalogue(
        table_relationships=[TableRelationship(table_a="orders", table_b="customers", join_key="customer_id")]
    ))
    assert "## Table Relationships" in md


def test_table_relationships_section_absent_when_none():
    md = _render_overview_md(_catalogue(table_relationships=[]))
    assert "## Table Relationships" not in md


def test_table_relationships_contains_mermaid_block():
    md = _render_overview_md(_catalogue(
        table_relationships=[TableRelationship(table_a="orders", table_b="customers", join_key="customer_id")]
    ))
    assert "```mermaid" in md


def test_table_relationships_contains_join_key():
    md = _render_overview_md(_catalogue(
        table_relationships=[TableRelationship(table_a="orders", table_b="customers", join_key="customer_id")]
    ))
    assert "customer_id" in md


def test_table_relationships_legend_present():
    md = _render_overview_md(_catalogue(
        table_relationships=[TableRelationship(table_a="a", table_b="b", join_key="id")]
    ))
    assert "Legend" in md


# ── tables reference section ──────────────────────────────────────────────────

def test_tables_reference_section_present():
    md = _render_overview_md(_catalogue())
    assert "## Tables Reference" in md


def test_table_name_in_tables_reference():
    md = _render_overview_md(_catalogue(tables=[_table("ast_assets")]))
    assert "ast_assets" in md


def test_table_description_in_tables_reference():
    md = _render_overview_md(_catalogue(
        tables=[_table("orders", description="One row per customer order.")]
    ))
    assert "One row per customer order." in md


def test_table_key_columns_in_tables_reference():
    md = _render_overview_md(_catalogue(
        tables=[_table("events", key_columns=["event_id", "occurred_at"])]
    ))
    assert "event_id" in md
    assert "occurred_at" in md


def test_per_table_data_quality_notes_shown():
    md = _render_overview_md(_catalogue(
        tables=[_table("orders", data_quality_notes=["null for 30% of rows."])]
    ))
    assert "null for 30% of rows." in md


def test_per_table_data_notes_absent_when_none():
    md = _render_overview_md(_catalogue(
        tables=[_table("orders", data_quality_notes=[])]
    ))
    assert "Data notes" not in md


# ── metrics section ───────────────────────────────────────────────────────────

def test_metrics_section_heading_present():
    md = _render_overview_md(_catalogue(metrics=[_metric("m", ["t"])]))
    assert "## Metrics" in md


def test_metric_name_as_bullet():
    md = _render_overview_md(_catalogue(
        metrics=[_metric("Monthly Revenue", ["t"])]
    ))
    assert "- **Monthly Revenue**" in md


def test_metric_frequency_label_present():
    md = _render_overview_md(_catalogue(
        metrics=[_metric("m", ["t"], granularity="quarterly")]
    ))
    assert "***Frequency:***" in md
    assert "quarterly" in md


def test_metric_unit_label_present():
    md = _render_overview_md(_catalogue(
        metrics=[_metric("m", ["t"], unit="€")]
    ))
    assert "***Unit:***" in md
    assert "€" in md


def test_metric_range_label_present():
    md = _render_overview_md(_catalogue(
        metrics=[_metric("m", ["t"], start="2022-03-01", end="2024-09-01")]
    ))
    assert "***Range:***" in md
    assert "2022-03" in md
    assert "2024-09" in md


def test_metric_description_present():
    md = _render_overview_md(_catalogue(
        metrics=[_metric("m", ["t"], description="Count of leads won each month.")]
    ))
    assert "Count of leads won each month." in md


def test_metric_sql_in_indented_code_block():
    sql = "SELECT strftime('%Y-%m-01', dt) AS month, COUNT(*) FROM leads GROUP BY 1"
    md = _render_overview_md(_catalogue(
        metrics=[_metric("m", ["t"], sql=sql)]
    ))
    assert "```sql" in md
    assert sql in md


def test_metric_tables_listed():
    md = _render_overview_md(_catalogue(
        tables=[_table("invoices"), _table("customers")],
        metrics=[_metric("m", tables=["invoices", "customers"])]
    ))
    assert "invoices" in md
    assert "customers" in md


def test_metric_grouped_by_group_field():
    md = _render_overview_md(_catalogue(
        tables=[_table("t")],
        metrics=[
            _metric("m1", ["t"], group="Revenue"),
            _metric("m2", ["t"], group="Accounts"),
        ]
    ))
    revenue_pos = md.index("### Revenue")
    accounts_pos = md.index("### Accounts")
    m1_pos = md.index("m1")
    m2_pos = md.index("m2")
    assert revenue_pos < m1_pos < accounts_pos
    assert accounts_pos < m2_pos


def test_metric_falls_back_to_primary_table_when_no_group():
    md = _render_overview_md(_catalogue(
        tables=[_table("orders"), _table("products")],
        metrics=[_metric("order_metric", tables=["orders"], group="")]
    ))
    assert "### orders" in md


def test_grouping_note_lists_all_groups():
    md = _render_overview_md(_catalogue(
        tables=[_table("t")],
        metrics=[
            _metric("m1", ["t"], group="Alpha"),
            _metric("m2", ["t"], group="Beta"),
        ]
    ))
    assert "**Alpha**" in md
    assert "**Beta**" in md


# ── facts section ─────────────────────────────────────────────────────────────

def test_facts_section_heading_present():
    md = _render_overview_md(_catalogue(facts=[_fact("f", ["t"])]))
    assert "## Facts" in md


def test_facts_explanation_note_present():
    md = _render_overview_md(_catalogue(facts=[_fact("f", ["t"])]))
    assert "not time-series" in md


def test_fact_name_as_heading():
    md = _render_overview_md(_catalogue(facts=[_fact("region_lookup", ["t"])]))
    assert "### region_lookup" in md


def test_fact_sql_in_code_block():
    sql = "SELECT region_id, region_name FROM regions"
    md = _render_overview_md(_catalogue(
        facts=[_fact("f", ["t"], sql=sql)]
    ))
    assert "```sql" in md
    assert sql in md


def test_fact_description_present():
    md = _render_overview_md(_catalogue(
        facts=[_fact("f", ["t"], description="All active regions.")]
    ))
    assert "All active regions." in md
