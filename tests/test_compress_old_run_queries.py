"""
Tests for compress_old_run_queries — strips bulky 'columns' and 'tables'
fields from old run_query tool calls to reduce context growth.

Each Phase 1 iteration adds one assistant message (tool calls) + one user
message (tool results). By iteration 43 on a large database, the columns
field alone from multi-table JOINs can add tens of thousands of tokens.

The last keep_last=2 iterations are left intact so the model has full context
on what it just did. Everything older has columns/tables stripped.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from schematica.backends import _AnthropicBackend, _LiteLLMBackend


# ── helpers ───────────────────────────────────────────────────────────────────

def _rq_block(tool_id: str, sql: str = "SELECT 1") -> SimpleNamespace:
    """Anthropic-format run_query tool_use block."""
    return SimpleNamespace(
        type="tool_use",
        name="run_query",
        id=tool_id,
        input={
            "sql": sql,
            "objective": "test objective",
            "plain_language": "count things",
            "columns": ["col_a", "col_b", "col_c"],
            "tables": ["Invoice", "InvoiceLine"],
        },
    )


def _rq_tool_call(tool_id: str, sql: str = "SELECT 1") -> dict:
    """LiteLLM-format run_query tool_call dict."""
    return {
        "id": tool_id,
        "type": "function",
        "function": {
            "name": "run_query",
            "arguments": json.dumps({
                "sql": sql,
                "objective": "test objective",
                "plain_language": "count things",
                "columns": ["col_a", "col_b", "col_c"],
                "tables": ["Invoice", "InvoiceLine"],
            }),
        },
    }


def _anthropic_backend_with_iterations(n: int) -> _AnthropicBackend:
    messages = [{"role": "user", "content": "schema..."}]
    for i in range(n):
        messages.append({
            "role": "assistant",
            "content": [_rq_block(f"t{i}")],
        })
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "result"}],
        })
    backend = _AnthropicBackend(None, "test-model", "system", messages)
    return backend


def _litellm_backend_with_iterations(n: int) -> _LiteLLMBackend:
    messages = [{"role": "user", "content": "schema..."}]
    for i in range(n):
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [_rq_tool_call(f"t{i}")],
        })
        messages.append({"role": "tool", "tool_call_id": f"t{i}", "content": "result"})
    backend = _LiteLLMBackend("test-model", "system", messages)
    return backend


# ── Anthropic backend ─────────────────────────────────────────────────────────

def test_anthropic_strips_columns_from_old_run_query():
    backend = _anthropic_backend_with_iterations(3)

    backend.compress_old_run_queries(keep_last=2)

    old_block = backend.messages[1]["content"][0]
    assert "columns" not in old_block.input


def test_anthropic_strips_tables_from_old_run_query():
    backend = _anthropic_backend_with_iterations(3)

    backend.compress_old_run_queries(keep_last=2)

    old_block = backend.messages[1]["content"][0]
    assert "tables" not in old_block.input


def test_anthropic_keeps_sql_in_old_run_query():
    backend = _anthropic_backend_with_iterations(3)

    backend.compress_old_run_queries(keep_last=2)

    old_block = backend.messages[1]["content"][0]
    assert "sql" in old_block.input


def test_anthropic_keeps_objective_in_old_run_query():
    backend = _anthropic_backend_with_iterations(3)

    backend.compress_old_run_queries(keep_last=2)

    old_block = backend.messages[1]["content"][0]
    assert "objective" in old_block.input


def test_anthropic_leaves_last_two_iterations_untouched():
    backend = _anthropic_backend_with_iterations(3)

    backend.compress_old_run_queries(keep_last=2)

    # iterations 2 and 3 (messages[3] and messages[5]) must still have columns
    recent_block_1 = backend.messages[3]["content"][0]
    recent_block_2 = backend.messages[5]["content"][0]
    assert "columns" in recent_block_1.input
    assert "columns" in recent_block_2.input


def test_anthropic_noop_when_fewer_iterations_than_keep_last():
    backend = _anthropic_backend_with_iterations(2)

    backend.compress_old_run_queries(keep_last=2)

    # Nothing should be stripped — all iterations are "recent"
    for msg in backend.messages:
        if msg.get("role") == "assistant":
            for block in msg["content"]:
                assert "columns" in block.input


def test_anthropic_does_not_touch_finish_catalogue_block():
    fc_block = SimpleNamespace(
        type="tool_use",
        name="finish_catalogue",
        id="fc1",
        input={"tables": ["Invoice"], "measurable_metrics": []},
    )
    messages = [
        {"role": "user", "content": "schema..."},
        {"role": "assistant", "content": [_rq_block("t0")]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "r"}]},
        {"role": "assistant", "content": [_rq_block("t1")]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "r"}]},
        {"role": "assistant", "content": [fc_block]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "fc1", "content": "ok"}]},
    ]
    backend = _AnthropicBackend(None, "test-model", "system", messages)

    backend.compress_old_run_queries(keep_last=2)

    assert "tables" in fc_block.input  # finish_catalogue tables field untouched


# ── LiteLLM backend ───────────────────────────────────────────────────────────

def test_litellm_strips_columns_from_old_run_query():
    backend = _litellm_backend_with_iterations(3)

    backend.compress_old_run_queries(keep_last=2)

    old_args = json.loads(backend.messages[1]["tool_calls"][0]["function"]["arguments"])
    assert "columns" not in old_args


def test_litellm_strips_tables_from_old_run_query():
    backend = _litellm_backend_with_iterations(3)

    backend.compress_old_run_queries(keep_last=2)

    old_args = json.loads(backend.messages[1]["tool_calls"][0]["function"]["arguments"])
    assert "tables" not in old_args


def test_litellm_keeps_sql_in_old_run_query():
    backend = _litellm_backend_with_iterations(3)

    backend.compress_old_run_queries(keep_last=2)

    old_args = json.loads(backend.messages[1]["tool_calls"][0]["function"]["arguments"])
    assert "sql" in old_args


def test_litellm_leaves_last_two_iterations_untouched():
    backend = _litellm_backend_with_iterations(3)

    backend.compress_old_run_queries(keep_last=2)

    recent_args_1 = json.loads(backend.messages[3]["tool_calls"][0]["function"]["arguments"])
    recent_args_2 = json.loads(backend.messages[5]["tool_calls"][0]["function"]["arguments"])
    assert "columns" in recent_args_1
    assert "columns" in recent_args_2


def test_litellm_noop_when_fewer_iterations_than_keep_last():
    backend = _litellm_backend_with_iterations(2)

    backend.compress_old_run_queries(keep_last=2)

    for msg in backend.messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
            assert "columns" in args


def test_litellm_does_not_touch_finish_catalogue_tool_call():
    fc_args = {"tables": ["Invoice"], "measurable_metrics": []}
    messages = [
        {"role": "user", "content": "schema..."},
        {"role": "assistant", "content": "", "tool_calls": [_rq_tool_call("t0")]},
        {"role": "tool", "tool_call_id": "t0", "content": "r"},
        {"role": "assistant", "content": "", "tool_calls": [_rq_tool_call("t1")]},
        {"role": "tool", "tool_call_id": "t1", "content": "r"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "fc1", "type": "function",
            "function": {"name": "finish_catalogue", "arguments": json.dumps(fc_args)},
        }]},
        {"role": "tool", "tool_call_id": "fc1", "content": "ok"},
    ]
    backend = _LiteLLMBackend("test-model", "system", messages)

    backend.compress_old_run_queries(keep_last=2)

    fc_stored = json.loads(messages[5]["tool_calls"][0]["function"]["arguments"])
    assert "tables" in fc_stored
