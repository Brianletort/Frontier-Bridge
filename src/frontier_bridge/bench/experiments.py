"""Experiment harnesses: offload sweep and context ladder.

These are short-probe experiment loops, not benchmarks: each point launches
llama-server, waits for health, runs brief probes, and tears down. Output is
JSON-lines, one record per point, suitable for committing under
results/experiments/. Interesting points get full `frontier bench` runs
afterwards.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from frontier_bridge.bench.client import stream_completion

PROBE_PROMPTS = [
    "Explain in two sentences why sparse mixture-of-experts models can run on less memory than their parameter count suggests.",
    "Write a Python one-liner that reverses the words in a string, then explain it briefly.",
]

FILLER = (
    "The quarterly infrastructure review covered rack density, cooling efficiency, "
    "interconnection latency, and capacity planning across the metro campuses. "
)
NEEDLE = "IMPORTANT: The migration codeword is BLUEHERON-42."
QUESTION = "Somewhere in the document above there is a migration codeword. What is it exactly?"
_FILLER_TOKENS_EST = 24  # rough tokens per filler sentence; fill ratio handles slack


def wait_health(port: int, process: subprocess.Popen, timeout_s: float) -> str:
    """Poll the server's /health endpoint until healthy, dead, or timed out."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return f"load_failed_exit_{process.returncode}"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return "healthy"
        except Exception:
            pass
        time.sleep(3)
    return "load_timeout"


def parse_metal_limits(log_path: Path) -> dict[str, Any]:
    """Pull Metal working-set numbers out of the server log when present."""
    info: dict[str, Any] = {"recommended_max_working_set_gb": None}
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return info
    match = re.search(r"recommendedMaxWorkingSetSize\s*=\s*(\d+(?:\.\d+)?)\s*MB", text)
    if match:
        info["recommended_max_working_set_gb"] = round(float(match.group(1)) / 1024, 1)
    return info


def parse_kv_size(log_path: Path) -> float | None:
    """Sum 'KV self size' / 'KV buffer size' MiB figures from the server log."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None
    sizes = [
        float(m) for m in re.findall(r"KV self size\s*=\s*([\d.]+)\s*MiB", text)
    ] or [
        float(m) for m in re.findall(r"KV buffer size\s*=\s*([\d.]+)\s*MiB", text)
    ]
    return round(sum(sizes), 1) if sizes else None


def build_needle_prompt(target_tokens: int) -> str:
    """Filler document with a needle at ~55% depth, ending with the question."""
    repeat = max(int(target_tokens / _FILLER_TOKENS_EST), 10)
    parts = []
    needle_at = int(repeat * 0.55)
    for i in range(repeat):
        parts.append(FILLER)
        if i == needle_at:
            parts.append(NEEDLE + " ")
    return "".join(parts) + "\n\n" + QUESTION


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def sweep_point(
    model: str, n_cpu_moe: int, ctx: int, port: int, extra: list[str]
) -> dict[str, Any]:
    """One offload-sweep point: launch, health, short decode probes, teardown."""
    log_path = Path(f"/tmp/sweep-ncpumoe-{n_cpu_moe}.log")
    cmd = [
        "llama-server", "-m", model, "-c", str(ctx), "-np", "1",
        "--host", "127.0.0.1", "--port", str(port),
        "-ngl", "999", "--n-cpu-moe", str(n_cpu_moe), "--jinja", *extra,
    ]
    record: dict[str, Any] = {
        "n_cpu_moe": n_cpu_moe,
        "ctx": ctx,
        "command": " ".join(cmd),
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": None,
        "load_seconds": None,
        "decode_tps": [],
        "ttft_ms": [],
        "errors": [],
    }
    with log_path.open("w") as log:
        started = time.monotonic()
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            status = wait_health(port, process, timeout_s=420)
            record["status"] = status
            if status != "healthy":
                return record
            record["load_seconds"] = round(time.monotonic() - started, 1)

            for prompt in PROBE_PROMPTS:
                timing = stream_completion(
                    port,
                    [{"role": "user", "content": prompt}],
                    prompt_id="probe",
                    max_tokens=96,
                    timeout_s=600,
                )
                if timing.error:
                    record["errors"].append(timing.error)
                else:
                    record["decode_tps"].append(timing.decode_tps)
                    record["ttft_ms"].append(timing.ttft_ms)
                if process.poll() is not None:
                    record["status"] = f"died_during_inference_exit_{process.returncode}"
                    return record
            record["status"] = "stable" if not record["errors"] else "errors_during_inference"
        finally:
            _terminate(process)
    record.update(parse_metal_limits(log_path))
    return record


def ladder_rung(
    model: str, ctx: int, n_cpu_moe: int, port: int, kv_quant: str | None
) -> dict[str, Any]:
    """One context-ladder rung: launch at ctx, measure KV footprint and a
    needle probe at ~70% of ctx, teardown."""
    log_path = Path(f"/tmp/ladder-ctx{ctx}-{kv_quant or 'f16'}.log")
    cmd = [
        "llama-server", "-m", model, "-c", str(ctx), "-np", "1",
        "--host", "127.0.0.1", "--port", str(port),
        "-ngl", "999", "--n-cpu-moe", str(n_cpu_moe), "--jinja",
    ]
    if kv_quant:
        cmd += ["-ctk", kv_quant, "-ctv", kv_quant]
    record: dict[str, Any] = {
        "ctx": ctx,
        "kv_quant": kv_quant or "f16",
        "n_cpu_moe": n_cpu_moe,
        "command": " ".join(cmd),
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": None,
        "kv_size_mib": None,
        "prompt_tokens": None,
        "ttft_ms": None,
        "prefill_tps": None,
        "decode_tps": None,
        "needle_found": None,
    }
    with log_path.open("w") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            status = wait_health(port, process, timeout_s=420)
            record["status"] = status
            if status != "healthy":
                return record
            record["kv_size_mib"] = parse_kv_size(log_path)

            prompt = build_needle_prompt(int(ctx * 0.7))
            timing = stream_completion(
                port,
                [{"role": "user", "content": prompt}],
                prompt_id=f"needle-{ctx}",
                max_tokens=120,
                timeout_s=3600,
            )
            if timing.error:
                record["status"] = f"probe_error: {timing.error}"
                if process.poll() is not None:
                    record["status"] = f"died_during_probe_exit_{process.returncode}"
                return record
            record.update(
                {
                    "status": "stable",
                    "prompt_tokens": timing.prompt_tokens,
                    "ttft_ms": timing.ttft_ms,
                    "prefill_tps": timing.prefill_tps,
                    "decode_tps": timing.decode_tps,
                    "needle_found": "BLUEHERON-42".lower() in timing.output_text.lower(),
                }
            )
        finally:
            _terminate(process)
    return record


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def kv_per_1k_tokens_mb(records: list[dict[str, Any]]) -> dict[str, float]:
    """Fold context-ladder records into measured KV MB per 1K tokens, per kv_quant.

    Only healthy rungs with a parsed KV size count. The per-rung ratio is
    kv_size_mib / (ctx/1000); rungs of the same kv_quant must agree (KV grows
    linearly with context), so the median is reported.
    """
    ratios: dict[str, list[float]] = {}
    for record in records:
        kv_mib = record.get("kv_size_mib")
        ctx = record.get("ctx")
        if record.get("status") != "stable" or not kv_mib or not ctx:
            continue
        quant = record.get("kv_quant") or "f16"
        ratios.setdefault(quant, []).append(kv_mib / (ctx / 1000))
    folded: dict[str, float] = {}
    for quant, values in ratios.items():
        ordered = sorted(values)
        folded[quant] = round(ordered[len(ordered) // 2], 1)
    return folded
