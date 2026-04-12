"""
Tests for pricing fallback coverage.

Every model that schematica may use must appear in the hardcoded fallback
table so get_model_pricing never silently returns wrong pricing.

When an unknown model is requested, get_model_pricing must warn rather than
silently returning a different model's rates.
"""
from __future__ import annotations

import warnings

import pytest

from schematica.pricing import _HARDCODED_FALLBACK, get_model_pricing


# ── all current-generation models have hardcoded entries ─────────────────────

EXPECTED_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

@pytest.mark.parametrize("model_id", EXPECTED_MODELS)
def test_model_has_hardcoded_fallback_entry(model_id):
    assert model_id in _HARDCODED_FALLBACK, (
        f"{model_id!r} is missing from _HARDCODED_FALLBACK — add it with correct pricing"
    )


# ── unknown model emits a warning instead of silently returning wrong rates ──

def test_unknown_model_emits_warning():
    with pytest.warns(UserWarning, match="No pricing data"):
        get_model_pricing("completely-unknown-model-xyz", pricing={})


def test_unknown_model_returns_a_dict_with_input_and_output_keys():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = get_model_pricing("completely-unknown-model-xyz", pricing={})
    assert "input" in result
    assert "output" in result
