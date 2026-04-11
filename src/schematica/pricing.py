"""
pricing.py — LLM cost formatting utilities.

Three-tier pricing lookup:
  1. Live fetch from LiteLLM GitHub JSON
  2. Cached result from last successful fetch (~/.cache/schematica/model_pricing.json)
  3. Hardcoded fallback table
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

_LITELLM_URL     = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
_LITELLM_TIMEOUT = 3  # seconds
_DEFAULT_CACHE_PATH = Path.home() / ".cache" / "schematica" / "model_pricing.json"

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
    "claude-sonnet-4-20250514":  _with_cache(3.00,  15.00),
    "claude-sonnet-4-6":         _with_cache(3.00,  15.00),
    "claude-haiku-4-5-20251001": _with_cache(0.80,   4.00),
    "claude-haiku-3-20240307":   _with_cache(0.25,   1.25),
}


def _fetch_from_litellm(url: str, timeout: int) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode())
        result = {}
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
                result[model_id] = entry
        return result or None
    except Exception:
        return None


def _load_cache(cache_path: Path) -> dict | None:
    try:
        with open(cache_path) as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(pricing: dict, cache_path: Path) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(pricing, f, indent=2)
    except Exception:
        pass


def build_pricing_table(
    url: str = _LITELLM_URL,
    cache_path: Path = _DEFAULT_CACHE_PATH,
    timeout: int = _LITELLM_TIMEOUT,
) -> tuple[dict, str]:
    live = _fetch_from_litellm(url, timeout)
    if live:
        _save_cache(live, cache_path)
        return live, "live"

    cached = _load_cache(cache_path)
    if cached:
        return cached, "cache"

    return dict(_HARDCODED_FALLBACK), "hardcoded"


def get_model_pricing(model_id: str, pricing: dict | None = None) -> dict:
    table = pricing if pricing is not None else MODEL_PRICING
    if model_id in table:
        return table[model_id]
    for key in table:
        if key.startswith(model_id):
            return table[key]
    return _HARDCODED_FALLBACK.get(model_id, {"input": 3.00, "output": 15.00})


MODEL_PRICING, PRICING_SOURCE = build_pricing_table()

_PRICING_SOURCE_LABELS = {
    "live":      "{cost}  (live pricing)",
    "cache":     "{cost}  ⚠ cached pricing — may be outdated, live fetch failed",
    "hardcoded": "{cost}  ⚠ hardcoded pricing — no cache yet, update pricing.py if rates changed",
}


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
