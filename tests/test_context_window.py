"""
Tests for get_context_window — look up max input tokens for a model.

Three-tier resolution (same pattern as pricing):
  1. Data extracted from LiteLLM's JSON during the live/cached fetch
  2. Static fallback table for models that are missing or have null values
     in LiteLLM's JSON (e.g. Anthropic models dropped from the plain-key map)
  3. Returns 0 for completely unknown models (callers treat 0 as "don't show")
"""
from __future__ import annotations

import pytest

from schematica.pricing import get_context_window


# ── known models resolved from the live/cached LiteLLM table ──────────────────

def test_gpt4o_returns_nonzero():
    # gpt-4o is well-covered in LiteLLM's JSON
    result = get_context_window("gpt-4o")
    assert result > 0


def test_gpt4o_returns_128k():
    assert get_context_window("gpt-4o") == 128_000


# ── static fallback for models with null/missing data in LiteLLM ──────────────

def test_anthropic_prefixed_claude_3_5_sonnet_returns_200k():
    assert get_context_window("anthropic/claude-3-5-sonnet-20241022") == 200_000


def test_bare_claude_3_5_sonnet_returns_200k():
    assert get_context_window("claude-3-5-sonnet-20241022") == 200_000


# ── unknown model returns 0 ────────────────────────────────────────────────────

def test_completely_unknown_model_returns_zero():
    assert get_context_window("unknown-vendor/unknown-model-xyz") == 0


# ── return type ───────────────────────────────────────────────────────────────

def test_returns_int():
    result = get_context_window("gpt-4o")
    assert isinstance(result, int)
