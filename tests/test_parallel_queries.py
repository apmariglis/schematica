"""
Tests for parallel run_query execution within a single agent iteration.

When the LLM returns multiple run_query tool calls in one response, they are
executed concurrently using a ThreadPoolExecutor.  The key invariants are:

  - All queries complete and their results are returned.
  - Results are in the same order as the input blocks (tool-id integrity).
  - Concurrent SELECT queries against a shared engine do not interfere.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from schematica.agent import _execute_query, _run_queries_parallel


@pytest.fixture()
def engine():
    # StaticPool shares a single in-memory connection across threads so all
    # threads see the same table data (default pool creates isolated in-memory
    # DBs per connection).
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE nums (n INTEGER)"))
        conn.execute(text("INSERT INTO nums VALUES (10), (20), (30)"))
    return eng


# ── _execute_query is thread-safe when called concurrently ────────────────────

def test_concurrent_queries_each_return_their_own_result(engine):
    # Three queries that return distinct values — results must match their query
    queries = ["SELECT 1 AS n", "SELECT 2 AS n", "SELECT 3 AS n"]

    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(lambda q: _execute_query(engine, q, "test"), queries))

    assert "1" in results[0]
    assert "2" in results[1]
    assert "3" in results[2]


def test_concurrent_queries_do_not_mix_rows(engine):
    # Both queries read from the same table but with different filters.
    q1 = "SELECT n FROM nums WHERE n < 15"
    q2 = "SELECT n FROM nums WHERE n > 25"

    with ThreadPoolExecutor(max_workers=2) as ex:
        r1, r2 = list(ex.map(lambda q: _execute_query(engine, q, ""), [q1, q2]))

    assert "10" in r1
    assert "20" not in r1
    assert "30" in r2
    assert "10" not in r2


# ── _run_queries_parallel — ordering and correctness ─────────────────────────

# A minimal stand-in for an Anthropic tool-use block
class _Block:
    def __init__(self, block_id: str, sql: str):
        self.id = block_id
        self.name = "run_query"
        self.input = {"sql": sql, "reason": "test", "tables": [], "columns": [], "plain_language": ""}


def test_parallel_results_preserve_input_order(engine):
    # Three queries — results must come back matched to the correct block id
    blocks = [
        _Block("id-a", "SELECT 1 AS n"),
        _Block("id-b", "SELECT 2 AS n"),
        _Block("id-c", "SELECT 3 AS n"),
    ]

    results = _run_queries_parallel(engine, blocks)

    assert results[0][0] == "id-a"
    assert results[1][0] == "id-b"
    assert results[2][0] == "id-c"


def test_parallel_results_contain_query_output(engine):
    blocks = [
        _Block("id-1", "SELECT 10 AS val"),
        _Block("id-2", "SELECT 20 AS val"),
    ]

    results = _run_queries_parallel(engine, blocks)

    assert "10" in results[0][1]
    assert "20" in results[1][1]


def test_parallel_single_query_works(engine):
    blocks = [_Block("only", "SELECT 42 AS answer")]

    results = _run_queries_parallel(engine, blocks)

    assert len(results) == 1
    assert results[0][0] == "only"
    assert "42" in results[0][1]


def test_parallel_empty_blocks_returns_empty_list(engine):
    results = _run_queries_parallel(engine, [])

    assert results == []


