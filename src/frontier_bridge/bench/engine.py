"""Assemble benchresult/v1 documents from a prompt-suite run.

The verified label is enforced by tooling, not discipline:
- status starts as "claimed";
- `finalize_status` upgrades to "verified" only when all four pins are non-null
  AND the result lists two reproductions.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

from frontier_bridge.bench.client import CompletionTiming, percentile, stream_completion

SUITES = ("chat", "coding_agent", "long_context", "tool_calling")


def load_suite(name_or_path: str) -> list[dict[str, Any]]:
    """Load a bundled suite by name, or any JSONL file by path."""
    if name_or_path in SUITES:
        ref = resources.files("frontier_bridge") / "bench" / "prompts" / f"{name_or_path}.jsonl"
        text = ref.read_text(encoding="utf-8")
    else:
        text = Path(name_or_path).read_text(encoding="utf-8")
    prompts = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [_expand_prompt(p) for p in prompts]


def _expand_prompt(prompt: dict[str, Any]) -> dict[str, Any]:
    """Expand a build_long_context spec into real messages (keeps JSONL small)."""
    spec = prompt.get("build_long_context")
    if not spec:
        return prompt
    repeat = int(spec.get("repeat", 100))
    filler = spec.get("filler", "")
    needle = spec.get("needle", "")
    position = float(spec.get("needle_position", 0.5))
    needle_at = int(repeat * position)
    parts = []
    for i in range(repeat):
        parts.append(filler)
        if needle and i == needle_at:
            parts.append(needle + " ")
    document = "".join(parts)
    expanded = dict(prompt)
    expanded.pop("build_long_context", None)
    expanded["messages"] = [
        {"role": "user", "content": f"{document}\n\n{spec.get('question', '')}"}
    ]
    return expanded


def plan_hash(plan_data: dict[str, Any]) -> str:
    canonical = json.dumps(plan_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def check_task(prompt: dict[str, Any], timing: CompletionTiming) -> bool:
    """Pass/fail per the prompt's declared check. No check declared: pass = completed."""
    if timing.error:
        return False
    check = prompt.get("check")
    if not check:
        return bool(timing.output_text.strip())
    kind = check.get("type")
    if kind == "contains_any":
        return any(v.lower() in timing.output_text.lower() for v in check.get("values", []))
    if kind == "contains_all":
        return all(v.lower() in timing.output_text.lower() for v in check.get("values", []))
    if kind == "min_tokens":
        produced = timing.completion_tokens or len(timing.inter_token_ms) + 1
        return produced >= int(check.get("value", 1))
    return bool(timing.output_text.strip())


def run_suite(
    port: int,
    suite_name: str,
    prompts: list[dict[str, Any]],
    echo=lambda s: None,
    model: str | None = None,
) -> tuple[list[CompletionTiming], list[dict[str, Any]]]:
    timings: list[CompletionTiming] = []
    tasks: list[dict[str, Any]] = []
    for i, prompt in enumerate(prompts):
        prompt_id = prompt.get("id", f"{suite_name}-{i}")
        echo(f"  [{i + 1}/{len(prompts)}] {prompt_id} ...")
        timing = stream_completion(
            port,
            prompt["messages"],
            prompt_id=prompt_id,
            max_tokens=prompt.get("max_tokens", 512),
            model=model,
        )
        timings.append(timing)
        passed = check_task(prompt, timing)
        tasks.append(
            {
                "task": prompt_id,
                "passed": passed,
                "notes": timing.error,
            }
        )
        echo(
            f"      ttft={timing.ttft_ms}ms decode={timing.decode_tps}tps "
            f"passed={passed}" + (f" error={timing.error}" if timing.error else "")
        )
    return timings, tasks


def build_result(
    result_id: str,
    suite_name: str,
    timings: list[CompletionTiming],
    tasks: list[dict[str, Any]],
    telemetry: dict[str, Any],
    pins: dict[str, Any],
    context_len: int | None,
    reproductions: list[str] | None = None,
    subject: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ok = [t for t in timings if not t.error]
    all_inter_token = [ms for t in ok for ms in t.inter_token_ms]
    ttfts = [t.ttft_ms for t in ok if t.ttft_ms is not None]
    decode_rates = [t.decode_tps for t in ok if t.decode_tps is not None]
    prefill_rates = [t.prefill_tps for t in ok if t.prefill_tps is not None]

    def _avg(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 2) if values else None

    result = {
        "schema_version": "benchresult/v1",
        "result_id": result_id,
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pins": pins,
        "metrics": {
            "ttft_ms": _avg(ttfts),
            "prefill_tps": _avg(prefill_rates),
            "decode_tps": _avg(decode_rates),
            "token_latency_ms": {
                "p50": percentile(all_inter_token, 50),
                "p95": percentile(all_inter_token, 95),
                "p99": percentile(all_inter_token, 99),
            },
            "peak_vram_gb": telemetry.get("peak_vram_gb"),
            "peak_ram_gb": telemetry.get("peak_ram_gb"),
            "ssd_read_gbps": None,  # per-run rate needs runtime exposure; total below
            "ssd_total_read_gb": telemetry.get("ssd_total_read_gb"),
            "expert_cache_hit_rate": None,  # null until a runtime exposes it
            "context_len_tokens": context_len,
        },
        "workload_tasks": tasks,
        "environment": {
            "os": f"{platform.system()} {platform.release()}",
            "notes": f"suite={suite_name}; client-side SSE chunk timing; "
            f"power_w_avg={telemetry.get('power_w_avg')}",
        },
        "reproductions": reproductions or [],
        "status": "claimed",
        "subject": subject or {"modelprofile": None, "workload": suite_name},
    }
    result = finalize_status(result)
    result["usability_suggested"] = suggest_usability(result, suite_name)
    return result


def finalize_status(result: dict[str, Any]) -> dict[str, Any]:
    """Upgrade to verified only with all four pins non-null + two reproductions."""
    pins = result.get("pins") or {}
    pinned = all(
        pins.get(key)
        for key in ("plan_hash", "model_sha256", "runtime_commit", "hwprofile_id")
    )
    reproduced = len(result.get("reproductions") or []) >= 2
    result["status"] = "verified" if (pinned and reproduced) else "claimed"
    return result


# Documented usability thresholds (v1). The tool suggests; maintainers decide.
_USABILITY_MIN_PASS_RATE = 0.8
_INTERACTIVE_MAX_P95_MS = 250.0
_INTERACTIVE_MIN_DECODE_TPS = 5.0
_AGENT_SUITES = ("coding_agent", "tool_calling")


def suggest_usability(result: dict[str, Any], suite_name: str) -> str:
    """Rules-based usability suggestion from measured metrics.

    unrated -> runs -> usable -> interactive -> agent_capable, with
    not_recommended when nothing completed. Thresholds are documented above;
    they are deliberately conservative and versioned with the code.
    """
    tasks = result.get("workload_tasks") or []
    metrics = result.get("metrics") or {}
    if not tasks:
        return "unrated"
    completed = [t for t in tasks if not (t.get("notes") or "").strip()]
    if not completed:
        return "not_recommended"
    rating = "runs"
    pass_rate = sum(1 for t in tasks if t.get("passed")) / len(tasks)
    if pass_rate >= _USABILITY_MIN_PASS_RATE:
        rating = "usable"
        p95 = (metrics.get("token_latency_ms") or {}).get("p95")
        decode = metrics.get("decode_tps")
        if (
            p95 is not None
            and decode is not None
            and p95 <= _INTERACTIVE_MAX_P95_MS
            and decode >= _INTERACTIVE_MIN_DECODE_TPS
        ):
            rating = "interactive"
            if suite_name in _AGENT_SUITES:
                rating = "agent_capable"
    return rating
