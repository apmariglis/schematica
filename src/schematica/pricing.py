"""
pricing.py — LLM cost formatting utilities.

Three-tier pricing lookup:
  1. Live fetch from LiteLLM GitHub JSON
  2. Cached result from last successful fetch (~/.cache/schematica/model_pricing.json)
  3. Hardcoded fallback table
"""
from __future__ import annotations

import json
import os
import urllib.request
import warnings
from pathlib import Path

_LITELLM_URL     = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
_LITELLM_TIMEOUT = 3  # seconds

def _default_cache_path() -> Path:
    """Return the cache path, respecting $XDG_CACHE_HOME if set."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "schematica" / "model_pricing.json"

_DEFAULT_CACHE_PATH = _default_cache_path()

CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER  = 0.10


def _with_cache(input: float, output: float) -> dict:
    return {
        "input":       input,
        "output":      output,
        "cache_write": round(input * CACHE_WRITE_MULTIPLIER, 6),
        "cache_read":  round(input * CACHE_READ_MULTIPLIER,  6),
    }


_HARDCODED_FALLBACK: dict[str, dict] = {
    # Prices in USD per million tokens — last verified 2025-05
    # Live fetch from LiteLLM supersedes these values at runtime.
    "claude-opus-4-6":           _with_cache(15.00,  75.00),
    "claude-sonnet-4-20250514":  _with_cache( 3.00,  15.00),
    "claude-sonnet-4-6":         _with_cache( 3.00,  15.00),
    "claude-haiku-4-5-20251001": _with_cache( 0.80,   4.00),
    "claude-haiku-3-20240307":   _with_cache( 0.25,   1.25),
    # Together AI — not always present in LiteLLM's live data
    "meta-llama/Llama-3.1-405B-Instruct-Turbo": {"input": 3.50, "output": 3.50},
    "meta-llama/Llama-3.3-70B-Instruct-Turbo":  {"input": 0.88, "output": 0.88},
}


def _fetch_from_litellm(url: str, timeout: int) -> dict | None:
    """Fetch LiteLLM's JSON and extract pricing + context windows in one pass.

    Returns {"pricing": {...}, "context_windows": {...}} or None on failure.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode())
        pricing: dict = {}
        context_windows: dict = {}
        for model_id, info in raw.items():
            if not isinstance(info, dict):
                continue
            inp = info.get("input_cost_per_token")
            out = info.get("output_cost_per_token")
            if inp is not None and out is not None:
                entry: dict = {
                    "input":  round(inp * 1_000_000, 6),
                    "output": round(out * 1_000_000, 6),
                }
                cw = info.get("cache_creation_input_token_cost")
                cr = info.get("cache_read_input_token_cost")
                if cw is not None:
                    entry["cache_write"] = round(cw * 1_000_000, 6)
                if cr is not None:
                    entry["cache_read"] = round(cr * 1_000_000, 6)
                pricing[model_id] = entry
            max_input = info.get("max_input_tokens")
            if max_input is not None:
                context_windows[model_id] = int(max_input)
        if not pricing:
            return None
        return {"pricing": pricing, "context_windows": context_windows}
    except Exception:
        return None


def _load_cache(cache_path: Path) -> dict | None:
    try:
        with open(cache_path) as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(data: dict, cache_path: Path) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        warnings.warn(
            f"Could not write pricing cache to {cache_path}: {exc}. "
            "Live pricing will be fetched on every run. "
            "Set $XDG_CACHE_HOME to a writable directory to fix this.",
            UserWarning,
            stacklevel=2,
        )


def _extract_pricing(data: dict) -> dict:
    """Extract pricing table from cache data, handling old (flat) and new (nested) formats."""
    return data.get("pricing", data) if "pricing" in data else data


def _extract_context_windows(data: dict) -> dict:
    """Extract context window table from cache data."""
    return data.get("context_windows", {})


def build_pricing_table(
    url: str = _LITELLM_URL,
    cache_path: Path = _DEFAULT_CACHE_PATH,
    timeout: int = _LITELLM_TIMEOUT,
) -> tuple[dict, str]:
    live = _fetch_from_litellm(url, timeout)
    if live:
        _save_cache(live, cache_path)
        return _extract_pricing(live), "live"

    cached = _load_cache(cache_path)
    if cached:
        return _extract_pricing(cached), "cache"

    return dict(_HARDCODED_FALLBACK), "hardcoded"


def build_context_window_table(
    url: str = _LITELLM_URL,
    cache_path: Path = _DEFAULT_CACHE_PATH,
    timeout: int = _LITELLM_TIMEOUT,
) -> dict:
    live = _fetch_from_litellm(url, timeout)
    if live:
        _save_cache(live, cache_path)
        return _extract_context_windows(live)

    cached = _load_cache(cache_path)
    if cached:
        return _extract_context_windows(cached)

    return {}


def get_model_pricing(model_id: str, pricing: dict | None = None) -> dict:
    # Strip provider prefix (e.g. "anthropic/claude-x" → "claude-x") so lookups
    # work regardless of whether the caller passes a prefixed or bare model name.
    bare_id = model_id.split("/", 1)[1] if "/" in model_id else model_id
    table = pricing if pricing is not None else MODEL_PRICING
    for lookup in (model_id, bare_id):
        if lookup in table:
            return table[lookup]
        for key in table:
            if key.startswith(lookup):
                return table[key]
    if bare_id in _HARDCODED_FALLBACK:
        return _HARDCODED_FALLBACK[bare_id]
    warnings.warn(
        f"No pricing data found for model {model_id!r} — cost estimate will be inaccurate. "
        "Add the model to _HARDCODED_FALLBACK in pricing.py or ensure a live/cached pricing "
        "table is available.",
        UserWarning,
        stacklevel=2,
    )
    return {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30}


MODEL_PRICING, PRICING_SOURCE = build_pricing_table()

# Context windows — populated from the same LiteLLM JSON fetch.
# Models with null or missing max_input_tokens in LiteLLM are covered by the
# fallback table below.
_CONTEXT_WINDOW_FALLBACK: dict[str, int] = {
    # OpenAI — well-known, stable values
    "gpt-4o":              128_000,
    "gpt-4o-mini":         128_000,
    "gpt-4-turbo":         128_000,
    "gpt-4":               128_000,
    "gpt-3.5-turbo":        16_385,
    "o1":                  200_000,
    "o1-mini":             128_000,
    "o3":                  200_000,
    "o3-mini":             200_000,
    # Anthropic — dropped from LiteLLM 1.83.0 plain-key map
    "anthropic/claude-3-5-sonnet-20241022":  200_000,
    "anthropic/claude-3-5-haiku-20241022":   200_000,
    "anthropic/claude-3-opus-20240229":      200_000,
    "anthropic/claude-3-sonnet-20240229":    200_000,
    "anthropic/claude-3-haiku-20240307":     200_000,
    "anthropic/claude-3-7-sonnet-20250219":  200_000,
    "claude-3-5-sonnet-20241022":            200_000,
    "claude-3-5-haiku-20241022":             200_000,
    "claude-3-7-sonnet-20250219":            200_000,
    # Together AI — stored in LiteLLM JSON but max_input_tokens is null
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo":        131_072,
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free":   131_072,
    "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo":    131_072,
    "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo":   131_072,
    "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo":  131_072,
}

CONTEXT_WINDOWS: dict[str, int] = build_context_window_table()

_PRICING_SOURCE_LABELS = {
    "live":      "{cost}  (live pricing)",
    "cache":     "{cost}  ⚠ cached pricing — may be outdated, live fetch failed",
    "hardcoded": "{cost}  ⚠ hardcoded pricing — no cache yet, update pricing.py if rates changed",
}


def get_context_window(model_id: str, context_windows: dict | None = None) -> int:
    """Return the context-window size (max input tokens) for a model, or 0 if unknown.

    Resolution order:
      1. LiteLLM JSON data (live-fetched or cached) via CONTEXT_WINDOWS
      2. Static fallback table for models with null/missing data in LiteLLM
      3. 0 — callers should treat this as "unknown, don't display"
    """
    bare_id = model_id.split("/", 1)[1] if "/" in model_id else model_id
    table = context_windows if context_windows is not None else CONTEXT_WINDOWS
    for lookup in (model_id, bare_id):
        if lookup in table:
            return table[lookup]
        if lookup in _CONTEXT_WINDOW_FALLBACK:
            return _CONTEXT_WINDOW_FALLBACK[lookup]
    return 0


def format_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> str:
    """Return a human-readable cost string for a completed API call."""
    pricing = get_model_pricing(model)
    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    if cache_creation_tokens:
        cost += cache_creation_tokens * pricing.get("cache_write", pricing["input"] * CACHE_WRITE_MULTIPLIER) / 1_000_000
    if cache_read_tokens:
        cost += cache_read_tokens * pricing.get("cache_read", pricing["input"] * CACHE_READ_MULTIPLIER) / 1_000_000
    cost_str = f"${cost:.4f}"
    template = _PRICING_SOURCE_LABELS.get(PRICING_SOURCE, "{cost}")
    return template.format(cost=cost_str)
