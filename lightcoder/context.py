"""Layered context compression.

When the running transcript approaches the model's window, we reclaim space in
escalating layers. The whole subsystem is gated by ``AgentConfig.compression``:
turn it off and the agent simply accumulates history until the model errors — which
is exactly the comparison the ablation wants to measure.

Layers (cheap to expensive):
  1. snip  — truncate the body of long, old tool outputs to head+tail lines.
  2. summarize — LLM-compress the older turns into a short briefing, keep recent intact.
  3. collapse — last resort: drop everything but a summary + the last few messages.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLM


def approx_tokens(text: str) -> int:
    """Estimate the tokens in *text*, counting ASCII and non-ASCII separately.

    One divisor cannot serve both scripts. Measured against a real tokenizer's reported
    ``prompt_tokens``, ASCII runs ~4 characters per token while CJK runs ~1.8, so the old
    ``len(text) // 3`` compromise overestimated English by up to 110% and — worse —
    *underestimated* Chinese by ~40%. Splitting the two roughly halves the error
    (mean absolute error 34.5% -> 21.6% over English prose, technical English, Chinese,
    Python, JSON and diffs).

    The remaining error is inherent to counting characters: real characters-per-token spans
    2.0 (diffs, punctuation-dense) to 6.3 (common English words) — a 3x range no fixed pair
    of constants can track. Symbol-dense content is still underestimated, which is the
    dangerous direction for a compaction trigger, so the layer thresholds below must keep
    enough headroom to absorb it.
    """
    ascii_chars = len(text.encode("ascii", "ignore"))   # C-speed ASCII count
    return math.ceil(ascii_chars / 4) + math.ceil((len(text) - ascii_chars) / 1.8)


def estimate_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        if m.get("content"):
            total += approx_tokens(m["content"])
        if m.get("tool_calls"):
            total += approx_tokens(str(m["tool_calls"]))
    return total


class ContextManager:
    def __init__(self, max_tokens: int = 128_000, enabled: bool = True):
        self.max_tokens = max_tokens
        self.enabled = enabled
        self._snip_at = int(max_tokens * 0.50)
        self._summarize_at = int(max_tokens * 0.70)
        self._collapse_at = int(max_tokens * 0.90)

    def maybe_compress(self, messages: list[dict], llm: "LLM | None" = None) -> bool:
        """Apply layers as needed. Returns True if anything was compressed.

        A no-op when compression is disabled (the ablation's 'off' arm)."""
        if not self.enabled:
            return False

        changed = False
        current = estimate_tokens(messages)

        if current > self._snip_at and self._snip_tool_outputs(messages):
            changed = True
            current = estimate_tokens(messages)

        if current > self._summarize_at and len(messages) > 10:
            if self._summarize_old(messages, llm, keep_recent=8):
                changed = True
                current = estimate_tokens(messages)

        if current > self._collapse_at and len(messages) > 4:
            self._collapse(messages, llm)
            changed = True

        return changed

    # --- layer 1 -----------------------------------------------------------
    @staticmethod
    def _snip_tool_outputs(messages: list[dict]) -> bool:
        changed = False
        for m in messages:
            if m.get("role") != "tool":
                continue
            content = m.get("content", "") or ""
            if len(content) <= 1500:
                continue
            lines = content.splitlines()
            if len(lines) <= 6:
                continue
            m["content"] = (
                "\n".join(lines[:3])
                + f"\n... ({len(lines)} lines, snipped to save context) ...\n"
                + "\n".join(lines[-3:])
            )
            changed = True
        return changed

    # --- layer 2 -----------------------------------------------------------
    def _summarize_old(self, messages: list[dict], llm, keep_recent: int) -> bool:
        if len(messages) <= keep_recent:
            return False
        old, tail = messages[:-keep_recent], messages[-keep_recent:]
        summary = self._summary(old, llm)
        messages.clear()
        messages.append(
            {"role": "user", "content": f"[Earlier conversation, compressed]\n{summary}"}
        )
        messages.append(
            {"role": "assistant", "content": "Understood — I have the earlier context."}
        )
        messages.extend(tail)
        return True

    # --- layer 3 -----------------------------------------------------------
    def _collapse(self, messages: list[dict], llm) -> None:
        tail = messages[-4:] if len(messages) > 4 else messages[-2:]
        summary = self._summary(messages[: -len(tail)], llm)
        messages.clear()
        messages.append({"role": "user", "content": f"[Context reset]\n{summary}"})
        messages.append(
            {"role": "assistant", "content": "Context restored. Continuing."}
        )
        messages.extend(tail)

    # --- summary helpers ---------------------------------------------------
    def _summary(self, messages: list[dict], llm) -> str:
        flat = self._flatten(messages)
        if llm is not None:
            try:
                resp = llm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Compress this conversation into a brief briefing. "
                                "Preserve: files edited, key decisions, errors hit, "
                                "and the current task state. Drop verbose output and "
                                "redundant back-and-forth."
                            ),
                        },
                        {"role": "user", "content": flat[:15000]},
                    ]
                )
                if resp.content.strip():
                    return resp.content
            except Exception:
                pass
        return self._extract(messages)

    @staticmethod
    def _flatten(messages: list[dict]) -> str:
        parts = []
        for m in messages:
            text = m.get("content", "") or ""
            if text:
                parts.append(f"[{m.get('role', '?')}] {text[:400]}")
        return "\n".join(parts)

    @staticmethod
    def _extract(messages: list[dict]) -> str:
        import re

        files, errors = set(), []
        for m in messages:
            text = m.get("content", "") or ""
            for match in re.finditer(r"[\w./\-]+\.\w{1,5}", text):
                files.add(match.group())
            for line in text.splitlines():
                if "error" in line.lower():
                    errors.append(line.strip()[:150])
        parts = []
        if files:
            parts.append(f"Files touched: {', '.join(sorted(files)[:20])}")
        if errors:
            parts.append(f"Errors seen: {'; '.join(errors[:5])}")
        return "\n".join(parts) or "(no extractable context)"
