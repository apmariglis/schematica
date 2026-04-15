"""
Tests for --model and --cache CLI flag interactions with SC_CACHE.

Bug (fixed): --model overrides MODEL after module-level _CACHE validation, so
running with SC_CACHE=true (anthropic in .env) + --model gemini/... created an
_AnthropicBackend with the old _ANTHROPIC_MODEL, not a _LiteLLMBackend for Gemini.

SC_CACHE defaults to false when absent — no need for a --no-cache flag since
the default state is already no-cache.

Expected behaviour:
  - --model non-anthropic               → _CACHE=False regardless of .env
  - --model anthropic (no flag)         → _CACHE unchanged from .env
  - --model anthropic + cache_override=True  → _CACHE=True
  - --model non-anthropic + cache_override=True → RuntimeError
"""
from __future__ import annotations

import pytest

import schematica.agent as agent_module


def _override(model: str, cache: "bool | None" = None) -> None:
    agent_module._apply_model_override(model, cache_override=cache)


def _setup(monkeypatch, model="anthropic/claude-haiku-4-5-20251001", cache=True):
    monkeypatch.setattr(agent_module, "_CACHE", cache)
    monkeypatch.setattr(agent_module, "MODEL", model)
    bare = model[len("anthropic/"):] if model.startswith("anthropic/") else model
    monkeypatch.setattr(agent_module, "_ANTHROPIC_MODEL", bare)


# ── non-anthropic override always disables cache ──────────────────────────────

def test_gemini_override_sets_cache_false(monkeypatch):
    _setup(monkeypatch)
    _override("gemini/gemini-2.5-flash")
    assert agent_module._CACHE is False


def test_together_ai_override_sets_cache_false(monkeypatch):
    _setup(monkeypatch)
    _override("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo")
    assert agent_module._CACHE is False


def test_non_anthropic_override_updates_model(monkeypatch):
    _setup(monkeypatch)
    _override("gemini/gemini-2.5-flash")
    assert agent_module.MODEL == "gemini/gemini-2.5-flash"


def test_non_anthropic_override_cache_already_false(monkeypatch):
    _setup(monkeypatch, model="gpt-4o", cache=False)
    _override("gemini/gemini-2.5-flash")
    assert agent_module._CACHE is False
    assert agent_module.MODEL == "gemini/gemini-2.5-flash"


# ── non-anthropic + explicit --cache → error ──────────────────────────────────

def test_non_anthropic_with_cache_true_raises(monkeypatch):
    _setup(monkeypatch)
    with pytest.raises(RuntimeError, match="cache"):
        _override("gemini/gemini-2.5-flash", cache=True)


# ── anthropic override with no cache flag keeps _CACHE unchanged ──────────────

def test_anthropic_override_no_flag_keeps_cache_true(monkeypatch):
    _setup(monkeypatch, cache=True)
    _override("anthropic/claude-sonnet-4-6")
    assert agent_module._CACHE is True


def test_anthropic_override_no_flag_keeps_cache_false(monkeypatch):
    _setup(monkeypatch, cache=False)
    _override("anthropic/claude-sonnet-4-6")
    assert agent_module._CACHE is False


def test_anthropic_override_updates_anthropic_model(monkeypatch):
    _setup(monkeypatch)
    _override("anthropic/claude-sonnet-4-6")
    assert agent_module._ANTHROPIC_MODEL == "claude-sonnet-4-6"


def test_anthropic_override_updates_model(monkeypatch):
    _setup(monkeypatch)
    _override("anthropic/claude-sonnet-4-6")
    assert agent_module.MODEL == "anthropic/claude-sonnet-4-6"


# ── anthropic + explicit --cache / --no-cache ─────────────────────────────────

def test_anthropic_with_cache_true_enables_cache(monkeypatch):
    # SC_CACHE=false in .env, but --cache on CLI forces it on
    _setup(monkeypatch, cache=False)
    _override("anthropic/claude-sonnet-4-6", cache=True)
    assert agent_module._CACHE is True


def test_anthropic_with_cache_true_still_updates_model(monkeypatch):
    _setup(monkeypatch, cache=False)
    _override("anthropic/claude-sonnet-4-6", cache=True)
    assert agent_module.MODEL == "anthropic/claude-sonnet-4-6"
    assert agent_module._ANTHROPIC_MODEL == "claude-sonnet-4-6"
