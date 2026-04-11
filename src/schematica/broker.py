"""
broker.py — Data Access Broker.

Reads a DataCatalogue JSON and executes metric queries against the database.

Usage:
    broker.fetch("monthly_installations_completed", "2022-01-01", "2024-12-31")

The broker:
    1. Finds the matching metric in the catalogue
    2. Executes the validated SQL query
    3. Filters to the requested date range
    4. Returns a normalised DataFrame with columns: date (Period), value (float)
"""
from __future__ import annotations

import json
import warnings
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from schematica.db import make_readonly_engine


class DataAccessBroker:
    """
    Executes catalogue metric queries against the database.

    Parameters
    ----------
    catalogue_path : str | Path
        Path to the data_catalogue.json produced by the Schematica.
    connection_string : str
        SQLAlchemy connection string for the same database the catalogue was
        generated from.
    """

    def __init__(self, catalogue_path: str | Path, connection_string: str) -> None:
        with open(catalogue_path) as f:
            raw = json.load(f)

        self._metrics: dict[str, dict] = {
            m["name"]: m for m in raw["measurable_metrics"]
        }
        self._facts: dict[str, dict] = {
            f["name"]: f for f in raw.get("queryable_facts", [])
        }
        self._engine = make_readonly_engine(connection_string)

    # ── public API ─────────────────────────────────────────────────────────────

    def list_metrics(self) -> list[str]:
        """Return all available metric names from the catalogue."""
        return sorted(self._metrics.keys())

    def fetch(
        self,
        metric_name: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch a metric as a normalised time-series DataFrame.

        Parameters
        ----------
        metric_name : str
            Metric name as it appears in the catalogue, or a close approximation
            (fuzzy matching is applied automatically).
        start_date : str | None
            ISO date string (YYYY-MM-DD). Rows before this date are excluded.
        end_date : str | None
            ISO date string (YYYY-MM-DD). Rows after this date are excluded.

        Returns
        -------
        pd.DataFrame
            Two columns: ``date`` (period string, e.g. "2022-01") and
            ``value`` (float). Sorted by date ascending.

        Raises
        ------
        KeyError
            If no metric can be matched to ``metric_name``.
        """
        resolved_name, score = self._find_metric(metric_name)
        if resolved_name is None:
            raise KeyError(
                f"No metric matching '{metric_name}' found in catalogue. "
                f"Available metrics: {self.list_metrics()}"
            )

        if score < 1.0:
            warnings.warn(
                f"Fuzzy match: '{metric_name}' resolved to '{resolved_name}' "
                f"(score={score:.2f}). Use the exact name to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )

        metric = self._metrics[resolved_name]
        df = self._execute(metric["sql"])

        # Standardise column names: first col → date, second col → value
        cols = list(df.columns)
        if len(cols) < 2:
            raise ValueError(
                f"Metric '{resolved_name}' query returned {len(cols)} column(s); "
                "expected at least 2 (date, value)."
            )
        df = df.rename(columns={cols[0]: "date", cols[1]: "value"})
        df = df[["date", "value"]].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df = df.sort_values("date").reset_index(drop=True)

        # Date range filtering
        if start_date:
            df = df[df["date"] >= start_date[:7]]  # compare YYYY-MM prefix
        if end_date:
            df = df[df["date"] <= end_date[:7]]

        df.attrs["metric_name"]  = resolved_name
        df.attrs["match_score"]  = round(score, 3)
        df.attrs["granularity"]  = metric.get("granularity", "unknown")
        df.attrs["unit"]         = metric.get("unit", "")
        df.attrs["confidence"]   = metric.get("confidence", "")
        df.attrs["agent_notes"]  = metric.get("agent_notes", "")

        return df

    def list_facts(self) -> list[str]:
        """Return all available queryable fact names from the catalogue."""
        return sorted(self._facts.keys())

    def query(self, fact_name: str) -> pd.DataFrame:
        """
        Fetch a queryable fact as a plain DataFrame.

        No column or shape normalisation is applied — the DataFrame reflects
        whatever columns the SQL returns. Fuzzy name matching is applied.

        Raises
        ------
        KeyError
            If no fact can be matched to ``fact_name``.
        """
        resolved_name, score = self._find_entry(fact_name, self._facts)
        if resolved_name is None:
            raise KeyError(
                f"No fact matching '{fact_name}' found in catalogue. "
                f"Available facts: {self.list_facts()}"
            )

        fact = self._facts[resolved_name]
        df = self._execute(fact["sql"])
        df.attrs["fact_name"]   = resolved_name
        df.attrs["match_score"] = round(score, 3)
        df.attrs["agent_notes"] = fact.get("agent_notes", "")
        return df

    def describe_fact(self, fact_name: str) -> dict:
        """Return the full catalogue entry for a queryable fact (or the best fuzzy match)."""
        resolved_name, _ = self._find_entry(fact_name, self._facts)
        if resolved_name is None:
            raise KeyError(f"No fact matching '{fact_name}' found.")
        return self._facts[resolved_name]

    def describe(self, metric_name: str) -> dict:
        """Return the full catalogue entry for a metric (or the best fuzzy match)."""
        resolved_name, _ = self._find_metric(metric_name)
        if resolved_name is None:
            raise KeyError(f"No metric matching '{metric_name}' found.")
        return self._metrics[resolved_name]

    # ── private ────────────────────────────────────────────────────────────────

    def _find_metric(self, query: str) -> tuple[str | None, float]:
        return self._find_entry(query, self._metrics)

    def _find_entry(self, query: str, entries: dict) -> tuple[str | None, float]:
        """
        Find the best matching entry name in the given dict.

        Priority:
        1. Exact match (case-insensitive)
        2. Best fuzzy match above threshold (0.4)
        """
        query_lower = query.lower().strip()

        for name in entries:
            if name.lower() == query_lower:
                return name, 1.0

        best_name, best_score = None, 0.0
        for name in entries:
            score = SequenceMatcher(None, query_lower, name.lower()).ratio()
            query_words = set(query_lower.replace("_", " ").split())
            name_words  = set(name.lower().replace("_", " ").split())
            overlap = len(query_words & name_words) / max(len(query_words | name_words), 1)
            combined = (score + overlap) / 2
            if combined > best_score:
                best_score = combined
                best_name  = name

        if best_score >= 0.4:
            return best_name, best_score

        return None, 0.0

    def _execute(self, sql: str) -> pd.DataFrame:
        with self._engine.connect() as conn:
            return pd.read_sql(text(sql), conn)
