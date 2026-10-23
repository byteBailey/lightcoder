"""LLM client — a thin wrapper over any OpenAI-compatible chat endpoint.

We only need three things from the model layer:
  1. send (messages, tools) and get back text and/or structured tool calls,
  2. a stable assistant-message dict to append to history,
  3. resilience to transient network/rate errors.

Local vLLM, DeepSeek, Kimi, OpenAI, Ollama — all expose the OpenAI schema, so
switching providers is just ``base_url`` + ``model``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from .config import LLMConfig


@dataclass
class ToolCall:
    """One tool invocation the model asked for."""

    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    message: dict = field(default_factory=dict)  # raw assistant msg for history
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""  # "stop" | "tool_calls" | "length" (truncated) | ...


# A max_tokens at/above this is treated as "no explicit cap": we omit it so the server
# generates up to (context_window - prompt). Sending a literal huge value would exceed
# max_model_len once any prompt is added and error out.
_UNCAPPED_MAX_TOKENS = 200_000


_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


class LLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.model = cfg.model
        self._client = OpenAI(
            api_key=cfg.api_key or "dummy",
            base_url=cfg.base_url,
            timeout=cfg.timeout,
            max_retries=0,  # we do our own retry loop with backoff
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
    ) -> LLMResponse:
        """One model call, streamed. ``on_token`` is invoked per content delta; the
        full content and any (delta-assembled) tool calls are returned in LLMResponse."""
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=self.cfg.temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        # Omit max_tokens when "uncapped" so vLLM uses all remaining context instead of
        # erroring on prompt+max_tokens > max_model_len.
        if self.cfg.max_tokens and self.cfg.max_tokens < _UNCAPPED_MAX_TOKENS:
            kwargs["max_tokens"] = self.cfg.max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = self._call_with_retry(kwargs)

        content_parts: list[str] = []
        tc_acc: dict[int, dict] = {}  # tool_call index -> {id, name, args}
        finish_reason = ""
        usage: dict = {}

        for chunk in stream:
            u = getattr(chunk, "usage", None)
            if u:
                usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(u, "total_tokens", 0) or 0,
                    "cached_tokens": _cached_tokens(u),
                }
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            delta = ch.delta
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                if on_token:
                    on_token(delta.content)
            for tc in getattr(delta, "tool_calls", None) or []:
                idx = tc.index if tc.index is not None else 0
                acc = tc_acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                if tc.id:
                    acc["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn:
                    if fn.name:
                        acc["name"] = fn.name
                    if fn.arguments:
                        acc["args"] += fn.arguments
            if ch.finish_reason:
                finish_reason = ch.finish_reason

        content = "".join(content_parts)
        ordered = [tc_acc[i] for i in sorted(tc_acc)]
        tool_calls = [
            ToolCall(id=a["id"], name=a["name"], arguments=_parse_args(a["args"]))
            for a in ordered
        ]

        message: dict = {"role": "assistant", "content": content}
        if ordered:
            message["tool_calls"] = [
                {
                    "id": a["id"],
                    "type": "function",
                    "function": {"name": a["name"], "arguments": a["args"]},
                }
                for a in ordered
            ]

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            message=message,
            usage=usage,
            finish_reason=finish_reason,
        )

    def _call_with_retry(self, kwargs: dict):
        delay = 1.0
        last_err = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except _RETRYABLE as e:  # transient — back off and retry
                last_err = e
                if attempt == self.cfg.max_retries:
                    break
                time.sleep(delay)
                delay = min(delay * 2, 16.0)
        raise RuntimeError(f"LLM call failed after retries: {last_err}") from last_err


def _cached_tokens(u) -> int:
    """Prompt tokens served from the provider's prefix cache, 0 when not reported.

    OpenAI and vLLM report ``prompt_tokens_details.cached_tokens``; DeepSeek reports a
    top-level ``prompt_cache_hit_tokens`` instead.
    """
    details = getattr(u, "prompt_tokens_details", None)
    n = getattr(details, "cached_tokens", None) if details is not None else None
    if n is None:
        n = getattr(u, "prompt_cache_hit_tokens", None)
    return int(n or 0)


def _parse_args(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {"value": val}
    except json.JSONDecodeError:
        # Some models emit not-quite-JSON; surface it rather than crashing the loop.
        return {"_raw": raw}
