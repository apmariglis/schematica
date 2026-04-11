"""
catalogue.py — Pydantic models for the Schematica output.

The DataCatalogue is the single output artefact of the agent.
It is written to JSON and can be consumed by downstream tools.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TimeRange(BaseModel):
    start: str = Field(description="ISO date string, e.g. 2022-01-01")
    end: str   = Field(description="ISO date string, e.g. 2024-12-31")


class MeasurableMetric(BaseModel):
    name: str = Field(
        description="snake_case metric identifier, e.g. monthly_installations_completed"
    )
    description: str = Field(
        description="Plain-English description of what this metric measures"
    )
    sql: str = Field(
        description=(
            "SQL query that returns exactly two columns: a date/period column and a numeric "
            "value column, ordered by date ascending. Must be valid for the target database dialect."
        )
    )
    time_range: TimeRange = Field(
        description="Earliest and latest date present in the query result"
    )
    granularity: Literal["daily", "weekly", "monthly", "quarterly", "annual", "tick"] = Field(
        description="Natural time granularity of the data"
    )
    unit: str = Field(
        description="Unit of the value column, e.g. count, €, hours, %, kW, ratio"
    )
    tables_used: list[str] = Field(
        description="Table names the query reads from"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "high = column names / types are unambiguous; "
            "medium = inferred from sample values or naming conventions; "
            "low = uncertain, agent made assumptions"
        )
    )
    agent_notes: str = Field(
        description="Reasoning behind column choices, any assumptions made, or data quality caveats"
    )

    @field_validator("sql", mode="before")
    @classmethod
    def strip_trailing_semicolon(cls, v: str) -> str:
        return v.strip().rstrip(";").strip() if isinstance(v, str) else v


class QueryableFact(BaseModel):
    name: str = Field(
        description="snake_case identifier, e.g. region_lookup or active_contract_snapshot"
    )
    description: str = Field(
        description="Plain-English description of what this fact represents"
    )
    sql: str = Field(
        description=(
            "SQL query returning the fact. No column or shape constraints — "
            "may return any number of rows and columns. "
            "Must be valid for the target database dialect."
        )
    )
    tables_used: list[str] = Field(
        default_factory=list,
        description="Table names the query reads from"
    )
    agent_notes: str = Field(
        description="How this fact was identified, any caveats, or freshness notes"
    )

    @field_validator("sql", mode="before")
    @classmethod
    def strip_trailing_semicolon(cls, v: str) -> str:
        return v.strip().rstrip(";").strip() if isinstance(v, str) else v


class TableSummary(BaseModel):
    name: str
    row_count: int
    description: str = Field(
        description="Agent's inference of what real-world entity or event this table represents"
    )
    key_columns: list[str] = Field(
        description="Columns most relevant for measurement (date cols, numeric metrics, status fields)"
    )


class DataCatalogue(BaseModel):
    analysed_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds"),
        description="ISO datetime when the catalogue was generated"
    )
    model: str      = Field(default="", description="LLM model used to generate this catalogue")
    connection: str = Field(description="Redacted connection string")
    dialect: str    = Field(description="Database dialect: sqlite / postgresql / mysql / ...")
    description: str = Field(
        default="",
        description=(
            "One or two sentences describing what domain or business this database covers "
            "and what kinds of questions it can answer."
        )
    )
    tables: list[TableSummary]
    measurable_metrics: list[MeasurableMetric]
    queryable_facts: list[QueryableFact] = Field(
        default_factory=list,
        description=(
            "Non-time-series data worth preserving: reference tables, static lookups, "
            "point-in-time snapshots. Accessible via DataAccessBroker.query()."
        )
    )
    time_coverage: TimeRange = Field(
        description="Overall earliest and latest date found across all tables"
    )
    data_quality_notes: list[str] = Field(
        description=(
            "Observations about data quality, gaps, sparse columns, legacy overlap, "
            "or anything a consumer of this catalogue should be aware of"
        )
    )
