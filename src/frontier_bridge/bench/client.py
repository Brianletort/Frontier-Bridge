"""Streaming OpenAI-compatible client that measures per-token timing.

TTFT and inter-token latencies are measured at SSE-chunk granularity from the
client side. That includes server scheduling and localhost transport — which is
what an agent actually experiences — and is recorded as such in the method
field.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompletionTiming:
    prompt_id: str
    ttft_ms: float | None = None
    inter_token_ms: list[float] = field(default_factory=list)
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    prefill_tps: float | None = None
    decode_tps: float | None = None
    output_text: str = ""
    error: str | None = None


def stream_completion(
    port: int,
    messages: list[dict[str, str]],
    prompt_id: str,
    max_tokens: int = 512,
    timeout_s: float = 900.0,
    model: str | None = None,
) -> CompletionTiming:
    """Run one streaming chat completion and record chunk-level timing.

    Some servers (LM Studio, vLLM) require the `model` field; llama.cpp
    ignores it. Pass the id reported by /v1/models.
    """
    timing = CompletionTiming(prompt_id=prompt_id)
    body: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if model:
        body["model"] = model
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    last_chunk_at: float | None = None
    chunks = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                now = time.perf_counter()
                usage = event.get("usage")
                if usage:
                    timing.completion_tokens = usage.get("completion_tokens")
                    timing.prompt_tokens = usage.get("prompt_tokens")
                choices = event.get("choices") or []
                delta_text = ""
                if choices:
                    delta_text = (choices[0].get("delta") or {}).get("content") or ""
                if not delta_text and not choices:
                    continue
                if timing.ttft_ms is None:
                    timing.ttft_ms = round((now - start) * 1000, 1)
                elif last_chunk_at is not None:
                    timing.inter_token_ms.append(round((now - last_chunk_at) * 1000, 2))
                last_chunk_at = now
                chunks += 1
                timing.output_text += delta_text
    except Exception as exc:  # noqa: BLE001 - a failed prompt is a result, not a crash
        timing.error = f"{type(exc).__name__}: {exc}"
        return timing

    total_s = time.perf_counter() - start
    tokens_out = timing.completion_tokens or chunks
    if timing.ttft_ms is not None:
        decode_s = total_s - timing.ttft_ms / 1000
        if decode_s > 0 and tokens_out > 1:
            timing.decode_tps = round((tokens_out - 1) / decode_s, 2)
        if timing.prompt_tokens and timing.ttft_ms > 0:
            timing.prefill_tps = round(timing.prompt_tokens / (timing.ttft_ms / 1000), 1)
    return timing


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(int(len(ordered) * pct / 100), len(ordered) - 1)
    return round(ordered[index], 2)
