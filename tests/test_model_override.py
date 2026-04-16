"""
Tests for --model and --cache CLI flag interactions with SC_CACHE.

Bug (fixed): --model overrides _config after module-level validation, so
running with SC_CACHE=true (anthropic in .env) + --model gemini/... created an
_AnthropicBackend with the old anthropic_model, not a _LiteLLMBackend for Gemini.

SC_CACHE defaults to false when absent — no need for a --no-cache flag since
the default state is already no-cache.

Expected behaviour:
  - --model non-anthropic               → _config.cache=False regardless of .env
  - --model anthropic (no flag)         → _config.cache unchanged from .env
  - --model anthropic + cache_override=True  → _config.cache=True
  - --model non-anthropic + cache_override=True → RuntimeError
"""
from __future__ import annotations

import pytest

from schematica import agent
from schematica.agent import _ModelConfig


def _override(model: str, cache: "bool | None" = None) -> None:
    agent._apply_model_override(model, cache_override=cache)


def _setup(monkeypatch, model="anthropic/claude-haiku-4-5-20251001", cache=True):
    monkeypatch.setattr(agent, "_config", _ModelConfig(model, cache))


# ── non-anthropic override always disables cache ──────────────────────────────

def test_gemini_override_sets_cache_false(monkeypatch):
    _setup(monkeypatch)
    _override("gemini/gemini-2.5-flash")
    assert agent._config.cache is False


def test_non_anthropic_provider_override_sets_cache_false(monkeypatch):
    _setup(monkeypatch)
    _override("gemini/gemini-2.5-flash")
    assert agent._config.cache is False


def test_non_anthropic_override_updates_model(monkeypatch):
    _setup(monkeypatch)
    _override("gemini/gemini-2.5-flash")
    assert agent._config.model == "gemini/gemini-2.5-flash"


def test_non_anthropic_override_cache_already_false(monkeypatch):
    _setup(monkeypatch, model="gpt-4o", cache=False)
    _override("gemini/gemini-2.5-flash")
    assert agent._config.cache is False
    assert agent._config.model == "gemini/gemini-2.5-flash"


# ── non-anthropic + explicit --cache → error ──────────────────────────────────

def test_non_anthropic_with_cache_true_raises(monkeypatch):
    _setup(monkeypatch)
    with pytest.raises(RuntimeError, match="cache"):
        _override("gemini/gemini-2.5-flash", cache=True)


# ── anthropic override with no cache flag keeps _CACHE unchanged ──────────────

def test_anthropic_override_no_flag_keeps_cache_true(monkeypatch):
    _setup(monkeypatch, cache=True)
    _override("anthropic/claude-sonnet-4-6")
    assert agent._config.cache is True


def test_anthropic_override_no_flag_keeps_cache_false(monkeypatch):
    _setup(monkeypatch, cache=False)
    _override("anthropic/claude-sonnet-4-6")
    assert agent._config.cache is False


def test_anthropic_override_updates_anthropic_model(monkeypatch):
    _setup(monkeypatch)
    _override("anthropic/claude-sonnet-4-6")
    assert agent._config.anthropic_model == "claude-sonnet-4-6"


def test_anthropic_override_updates_model(monkeypatch):
    _setup(monkeypatch)
    _override("anthropic/claude-sonnet-4-6")
    assert agent._config.model == "anthropic/claude-sonnet-4-6"


# ── anthropic + explicit --cache / --no-cache ─────────────────────────────────

def test_anthropic_with_cache_true_enables_cache(monkeypatch):
    # SC_CACHE=false in .env, but --cache on CLI forces it on
    _setup(monkeypatch, cache=False)
    _override("anthropic/claude-sonnet-4-6", cache=True)
    assert agent._config.cache is True


def test_anthropic_with_cache_true_still_updates_model(monkeypatch):
    _setup(monkeypatch, cache=False)
    _override("anthropic/claude-sonnet-4-6", cache=True)
    assert agent._config.model == "anthropic/claude-sonnet-4-6"
    assert agent._config.anthropic_model == "claude-sonnet-4-6"


# ── anthropic + explicit cache_override=False disables cache ──────────────────

def test_anthropic_override_explicit_cache_false_disables_cache(monkeypatch):
    # cache_override=False must turn caching off even when it was previously on.
    _setup(monkeypatch, cache=True)
    _override("anthropic/claude-sonnet-4-6", cache=False)
    assert agent._config.cache is False
