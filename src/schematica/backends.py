"""
backends.py — LLM backend adapters for the Schematica agent loop.

Two backends are supported:

  _AnthropicBackend  — native Anthropic SDK, used when SC_CACHE=true.
                       Maintains messages in Anthropic content-block format.

  _LiteLLMBackend    — LiteLLM (OpenAI-compatible), used for all other models.
                       Maintains messages in OpenAI chat format.

Both expose the same interface so the agent loop is backend-agnostic:
  .call(tools, max_tokens)          → raw provider response
  .extract_usage(response)          → dict of token counts
  .stop_reason(response)            → "tool_use" | "end_turn" | "max_tokens"
  .tool_calls(response)             → list of tool-use blocks
  .append_assistant(response)       → record assistant turn in history
  .append_tool_results(results)     → record tool results in history
  .append_orphaned_errors(response) → inject errors for unresolved tool calls
  .append_user(text)                → append a plain user message
  .compress_finish_catalogue(...)   → shrink a large finish_catalogue payload
  .compress_truncated()             → shrink last truncated assistant message
"""
from __future__ import annotations

import json
import os
import warnings
from types import SimpleNamespace

_INSTANCE_ID = os.getpid()  # unique per process — disambiguates concurrent runs
_empty_response_count = 0   # incremented each time the API returns empty choices


def _try_int(val) -> int | None:
    """Parse an integer from a header value string; return None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _write_empty_response_dump(
    n: int,
    model: str,
    system: str,
    messages: list,
    tools: list,
    prompt_tokens: int | None,
) -> None:
    """Write the full LiteLLM request to debug_empty_N.json for post-mortem inspection.

    A new numbered file is created for each empty response so every occurrence
    is preserved even when the model recovers after a nudge and later fails again.
    """
    path = f"debug_empty_{_INSTANCE_ID}_{n}.json"
    payload = {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "system": system,
        "messages": messages,
        "tools": tools,
    }
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as exc:
        warnings.warn(f"Could not write empty-response dump to {path}: {exc}", UserWarning, stacklevel=2)


class _AnthropicBackend:
    """Anthropic-native backend. Maintains messages in Anthropic format."""

    def __init__(self, client, model: str, system_prompt: str, messages: list, cache: bool = False):
        self._client = client
        self._model = model
        self._system = system_prompt
        self._cache = cache
        self.messages = messages  # shared, mutated in-place

    def call(self, tools: list, max_tokens: int):
        # Streaming is required for requests that may take longer than 10 minutes.
        # get_final_message() returns a Message object with the same interface as
        # the non-streaming create() response, so no other code needs to change.
        system_block = {"type": "text", "text": self._system}
        if self._cache:
            system_block["cache_control"] = {"type": "ephemeral"}
        with self._client.messages.stream(
            model=self._model,
            max_tokens=max_tokens,
            system=[system_block],
            messages=self.messages,
            tools=tools,
        ) as stream:
            msg = stream.get_final_message()
            h = stream.response.headers
            self._last_rl = {
                "limit":     _try_int(h.get("anthropic-ratelimit-output-tokens-limit")),
                "remaining": _try_int(h.get("anthropic-ratelimit-output-tokens-remaining")),
            }
            return msg

    def last_output_rate_limit(self) -> dict:
        """Return the output-token rate-limit headers from the most recent call.

        Returns a dict with 'limit' and 'remaining' (both int or None).
        """
        return getattr(self, "_last_rl", {})

    def extract_usage(self, response) -> dict:
        u = response.usage
        return {
            "input_tokens":          getattr(u, "input_tokens",                0),
            "output_tokens":         getattr(u, "output_tokens",               0),
            "cache_creation_tokens": getattr(u, "cache_creation_input_tokens", 0),
            "cache_read_tokens":     getattr(u, "cache_read_input_tokens",     0),
            "thinking_tokens":       0,
        }

    def stop_reason(self, response) -> str:
        return response.stop_reason  # "tool_use" | "end_turn" | "max_tokens"

    def tool_calls(self, response) -> list:
        return [b for b in response.content if getattr(b, "type", None) == "tool_use"]

    def append_assistant(self, response) -> None:
        self.messages.append({"role": "assistant", "content": response.content})

    def append_tool_results(self, results: list[tuple[str, str]]) -> None:
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": content}
            for tid, content in results
        ]})

    def append_orphaned_errors(self, response) -> bool:
        """Inject tool_results for any orphaned tool_use blocks. Returns True if any found."""
        orphaned = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        if not orphaned:
            return False
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": b.id,
             "content": "Response was truncated before this tool call completed. Please retry."}
            for b in orphaned
        ]})
        return True

    def append_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def compress_finish_catalogue(self, block_id: str, summary: dict) -> None:
        last = self.messages[-1]
        if last.get("role") != "assistant":
            return
        for block in last.get("content", []):
            if getattr(block, "type", None) == "tool_use" and block.id == block_id:
                block.input = summary
                return

    def compress_truncated(self) -> None:
        """Replace tool_use inputs in the last (truncated) assistant message with a tiny placeholder."""
        if not self.messages or self.messages[-1].get("role") != "assistant":
            return
        for block in self.messages[-1].get("content", []):
            if getattr(block, "type", None) == "tool_use" and isinstance(block.input, dict):
                block.input = {"_truncated": True, "keys": list(block.input.keys())}

    def compress_old_run_queries(self, keep_last: int = 2) -> None:
        """Strip 'columns' and 'tables' from old run_query tool calls to reduce context size.

        Keeps the last keep_last assistant messages (iterations) intact so the
        model has full context on what it just did.
        """
        assistant_indices = [
            i for i, m in enumerate(self.messages) if m.get("role") == "assistant"
        ]
        old_indices = assistant_indices[:-keep_last] if keep_last > 0 else assistant_indices
        for i in old_indices:
            for block in self.messages[i].get("content", []):
                if (getattr(block, "type", None) == "tool_use"
                        and getattr(block, "name", None) == "run_query"
                        and isinstance(block.input, dict)):
                    block.input.pop("columns", None)
                    block.input.pop("tables", None)


class _LiteLLMBackend:
    """LiteLLM backend. Maintains messages in OpenAI format."""

    def __init__(self, model: str, system_prompt: str, messages: list):
        self._model = model
        self._system = system_prompt
        self.messages = messages  # already in OpenAI format

    def call(self, tools: list, max_tokens: int):
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm is required for non-Anthropic providers. "
                "Install it with: uv sync --extra litellm"
            )
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name":        t["name"],
                    "description": t.get("description", ""),
                    "parameters":  t["input_schema"],
                },
            }
            for t in tools
        ]
        response = litellm.completion(
            model=self._model,
            messages=[{"role": "system", "content": self._system}] + self.messages,
            tools=openai_tools,
            max_tokens=max_tokens,
        )
        if not response.choices:
            global _empty_response_count
            _empty_response_count += 1
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            details = getattr(usage, "completion_tokens_details", None)
            thinking_tokens = getattr(details, "reasoning_tokens", None)
            _write_empty_response_dump(
                n=_empty_response_count,
                model=self._model,
                system=self._system,
                messages=self.messages,
                tools=openai_tools,
                prompt_tokens=prompt_tokens,
            )
            parts = []
            if prompt_tokens is not None:
                parts.append(f"prompt_tokens={prompt_tokens}")
            if output_tokens is not None:
                parts.append(f"output_tokens={output_tokens}")
            if thinking_tokens is not None:
                parts.append(f"thinking_tokens={thinking_tokens}")
            token_info = (", " + ", ".join(parts)) if parts else ""
            err = ValueError(
                f"LiteLLM returned empty choices (model={self._model}{token_info}). "
                "This usually means the response was filtered or the provider returned an error."
            )
            err.empty_response_tokens = {
                "prompt_tokens":   prompt_tokens,
                "output_tokens":   output_tokens,
                "thinking_tokens": thinking_tokens,
            }
            raise err
        return response

    def last_output_rate_limit(self) -> dict:
        """No-op for LiteLLM — output TPM headers are not reliably available."""
        return {}

    def extract_usage(self, response) -> dict:
        u = response.usage
        details = getattr(u, "completion_tokens_details", None)
        thinking_tokens = getattr(details, "reasoning_tokens", 0) or 0
        return {
            "input_tokens":          getattr(u, "prompt_tokens",     0),
            "output_tokens":         getattr(u, "completion_tokens", 0),
            "cache_creation_tokens": 0,
            "cache_read_tokens":     0,
            "thinking_tokens":       thinking_tokens,
        }

    def _choice(self, response):
        return response.choices[0]

    def stop_reason(self, response) -> str:
        reason = self._choice(response).finish_reason
        if reason == "length":      return "max_tokens"
        if reason == "tool_calls":  return "tool_use"
        return "end_turn"

    def tool_calls(self, response) -> list:
        tcs = self._choice(response).message.tool_calls or []
        result = []
        for tc in tcs:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            result.append(SimpleNamespace(
                type="tool_use",
                id=tc.id,
                name=tc.function.name,
                input=args,
            ))
        return result

    def append_assistant(self, response) -> None:
        msg = self._choice(response).message
        d: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        self.messages.append(d)

    def append_tool_results(self, results: list[tuple[str, str]]) -> None:
        for tid, content in results:
            self.messages.append({"role": "tool", "tool_call_id": tid, "content": content})

    def append_orphaned_errors(self, response) -> bool:
        tcs = self._choice(response).message.tool_calls or []
        if not tcs:
            return False
        for tc in tcs:
            self.messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": "Response was truncated before this tool call completed. Please retry.",
            })
        return True

    def append_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def compress_finish_catalogue(self, block_id: str, summary: dict) -> None:
        last = self.messages[-1]
        if last.get("role") != "assistant":
            return
        for tc in last.get("tool_calls", []):
            if isinstance(tc, dict) and tc.get("id") == block_id:
                tc["function"]["arguments"] = json.dumps(summary)
                return

    def compress_truncated(self) -> None:
        """Replace tool_call arguments in the last (truncated) assistant message with a tiny placeholder."""
        if not self.messages or self.messages[-1].get("role") != "assistant":
            return
        for tc in self.messages[-1].get("tool_calls", []):
            if not isinstance(tc, dict):
                continue
            args_str = tc.get("function", {}).get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                keys = list(args.keys()) if isinstance(args, dict) else []
            except (json.JSONDecodeError, TypeError):
                keys = []
            tc["function"]["arguments"] = json.dumps({"_truncated": True, "keys": keys})
        self.messages[-1]["content"] = ""

    def compress_old_run_queries(self, keep_last: int = 2) -> None:
        """Strip 'columns' and 'tables' from old run_query tool calls to reduce context size.

        Keeps the last keep_last assistant messages (iterations) intact so the
        model has full context on what it just did.
        """
        assistant_indices = [
            i for i, m in enumerate(self.messages) if m.get("role") == "assistant"
        ]
        old_indices = assistant_indices[:-keep_last] if keep_last > 0 else assistant_indices
        for i in old_indices:
            for tc in self.messages[i].get("tool_calls", []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function", {})
                if fn.get("name") != "run_query":
                    continue
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(args, dict):
                    continue
                args.pop("columns", None)
                args.pop("tables", None)
                fn["arguments"] = json.dumps(args)
