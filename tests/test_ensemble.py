"""
Tests for ensemble exploration — _format_ensemble_context.

The function formats N Phase-1 query logs into a single Phase-2 context
message. It must include the schema, every query from every run, and a clear
Phase-2 instruction.
"""
from __future__ import annotations

import pytest

from schematica.agent import _dedup_query_logs
from schematica.agent import _format_ensemble_context


SCHEMA = "TABLE orders (id INTEGER, created_at DATE, amount REAL)"

QUERY_LOG_A = [
    {
        "sql": "SELECT strftime('%Y-%m', created_at) AS m, COUNT(*) FROM orders GROUP BY m",
        "reason": "Validate monthly order count",
        "plain_language": "Monthly order count",
        "result": "m | COUNT(*)\n2024-01 | 120\n2024-02 | 95",
    },
    {
        "sql": "SELECT strftime('%Y-%m', created_at) AS m, SUM(amount) FROM orders GROUP BY m",
        "reason": "Validate monthly revenue",
        "plain_language": "Monthly revenue",
        "result": "m | SUM(amount)\n2024-01 | 9800.0\n2024-02 | 7200.0",
    },
]

QUERY_LOG_B = [
    {
        "sql": "SELECT MIN(created_at), MAX(created_at) FROM orders",
        "reason": "Check date range",
        "plain_language": "Date range of orders",
        "result": "MIN | MAX\n2023-01-01 | 2024-12-31",
    },
]


# ── schema is always present ──────────────────────────────────────────────────

def test_schema_included_in_output():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A])

    assert SCHEMA in result


# ── every query from every run appears ───────────────────────────────────────

def test_all_queries_from_single_run_included():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A])

    assert "Monthly order count" in result
    assert "Monthly revenue" in result


def test_queries_from_both_runs_included():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A, QUERY_LOG_B])

    assert "Monthly order count" in result
    assert "Date range of orders" in result


def test_sql_from_each_query_appears():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A])

    assert "SELECT strftime('%Y-%m', created_at) AS m, COUNT(*)" in result


def test_result_from_each_query_appears():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A])

    assert "2024-01 | 120" in result


# ── run labels ────────────────────────────────────────────────────────────────

def test_run_count_label_for_single_run():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A])

    assert "1 time" in result


def test_run_count_label_for_multiple_runs():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A, QUERY_LOG_B])

    assert "2 times" in result


def test_run_section_headers_present_for_two_runs():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A, QUERY_LOG_B])

    # New format: "— Run 1 / 2  (N queries) —"
    assert "Run 1 / 2" in result
    assert "Run 2 / 2" in result


def test_query_count_per_run_shown():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A, QUERY_LOG_B])

    assert "2 queries" in result  # run A has 2
    assert "1 queries" in result  # run B has 1


# ── phase 2 instruction ───────────────────────────────────────────────────────

def test_phase_2_instruction_present():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A])

    assert "PHASE 2" in result
    assert "finish_catalogue" in result


def test_run_query_not_available_stated():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A])

    assert "run_query tool is not available" in result


# ── result truncation ─────────────────────────────────────────────────────────

def test_long_result_is_truncated():
    long_result = "x" * 2000
    log = [{"sql": "SELECT 1", "reason": "test", "plain_language": "test", "result": long_result}]

    result = _format_ensemble_context(SCHEMA, [log])

    assert "truncated" in result
    assert long_result not in result  # full string must not appear


def test_short_result_is_not_truncated():
    short_result = "col\n123"
    log = [{"sql": "SELECT 1", "reason": "test", "plain_language": "test", "result": short_result}]

    result = _format_ensemble_context(SCHEMA, [log])

    assert short_result in result


# ── empty log edge case ───────────────────────────────────────────────────────

def test_empty_query_log_produces_valid_output():
    result = _format_ensemble_context(SCHEMA, [[]])

    assert SCHEMA in result
    assert "finish_catalogue" in result


def test_total_query_count_in_header():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A, QUERY_LOG_B])

    # QUERY_LOG_A has 2, QUERY_LOG_B has 1 → 3 total
    assert "3" in result


# ── phase1_catalogues proposals ───────────────────────────────────────────────

# A minimal valid catalogue proposal (subset of fields used by _format_ensemble_context)
CATALOGUE_PROPOSAL = {
    "measurable_metrics": [
        {"name": "New Orders", "sql": "SELECT date, COUNT(*) FROM orders GROUP BY date"},
        {"name": "Average Order Value", "sql": "SELECT date, AVG(amount) FROM orders GROUP BY date"},
    ],
    "queryable_facts": [
        {"name": "Total Order Count"},
    ],
}


def test_no_proposals_section_when_catalogues_not_provided():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A])

    assert "PHASE-1 CATALOGUE PROPOSALS" not in result


def test_no_proposals_section_when_all_catalogues_are_none():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A], phase1_catalogues=[None])

    assert "PHASE-1 CATALOGUE PROPOSALS" not in result


def test_proposals_section_present_when_catalogue_provided():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A], phase1_catalogues=[CATALOGUE_PROPOSAL])

    assert "PHASE-1 CATALOGUE PROPOSALS" in result


def test_metric_names_listed_in_proposals_section():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A], phase1_catalogues=[CATALOGUE_PROPOSAL])

    assert "New Orders" in result
    assert "Average Order Value" in result


def test_fact_names_listed_in_proposals_section():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A], phase1_catalogues=[CATALOGUE_PROPOSAL])

    assert "Total Order Count" in result


def test_proposal_count_reflects_only_non_none_entries():
    # 2 runs, but only 1 submitted a catalogue (the other is None)
    result = _format_ensemble_context(
        SCHEMA,
        [QUERY_LOG_A, QUERY_LOG_B],
        phase1_catalogues=[CATALOGUE_PROPOSAL, None],
    )

    assert "1 of 2" in result


def test_union_instruction_present_when_proposals_included():
    result = _format_ensemble_context(SCHEMA, [QUERY_LOG_A], phase1_catalogues=[CATALOGUE_PROPOSAL])

    assert "UNION" in result.upper()


# ── _dedup_query_logs ─────────────────────────────────────────────────────────

def test_dedup_removes_identical_sql_across_runs():
    # Same SQL appears in both run A and run B — only the first occurrence kept.
    shared = {"sql": "SELECT COUNT(*) FROM orders", "reason": "r", "plain_language": "pl", "result": "1"}
    log_a = [shared]
    log_b = [shared]

    result = _dedup_query_logs([log_a, log_b])

    assert sum(len(l) for l in result) == 1


def test_dedup_keeps_unique_sql_from_both_runs():
    q1 = {"sql": "SELECT COUNT(*) FROM orders", "reason": "r", "plain_language": "pl", "result": "1"}
    q2 = {"sql": "SELECT MAX(created_at) FROM orders", "reason": "r", "plain_language": "pl", "result": "x"}
    log_a = [q1]
    log_b = [q2]

    result = _dedup_query_logs([log_a, log_b])

    assert sum(len(l) for l in result) == 2


def test_dedup_preserves_run_grouping_structure():
    q1 = {"sql": "SELECT 1", "reason": "r", "plain_language": "pl", "result": "1"}
    q2 = {"sql": "SELECT 2", "reason": "r", "plain_language": "pl", "result": "2"}

    result = _dedup_query_logs([[q1], [q2]])

    assert len(result) == 2  # two runs preserved


def test_dedup_empty_logs_returns_empty_structure():
    result = _dedup_query_logs([[], []])

    assert result == [[], []]


def test_dedup_ignores_leading_trailing_whitespace_in_sql():
    q1 = {"sql": "SELECT 1", "reason": "r", "plain_language": "pl", "result": "1"}
    q2 = {"sql": "  SELECT 1  ", "reason": "r", "plain_language": "pl", "result": "1"}

    result = _dedup_query_logs([[q1], [q2]])

    # Both SQL strings normalise to "SELECT 1" → second is a duplicate
    assert sum(len(l) for l in result) == 1
