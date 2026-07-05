"""Runtime adapters v0: wrappers, not integrations.

Each adapter generates the exact launch command for an existing runtime.
Commands contain <PLACEHOLDER> tokens for values we cannot know (artifact
paths); everything else is explicit. Deep adapters (telemetry capture,
lifecycle management) come later. See runtime_adapters/README.md.

Verify flags against each runtime's current docs before running — runtimes
evolve faster than this table.
"""

from __future__ import annotations

from typing import Any

# Hardware class -> ordered runtime preference (from the runtime selection table).
_RUNTIME_TABLE: dict[str, list[str]] = {
    "apple_unified": ["ds4", "mlx", "llama_cpp"],
    "nvidia_gb10": ["ds4_zgx", "ds4", "llama_cpp"],
    "nvidia_device_local": ["llama_cpp", "ds4", "vllm", "sglang"],
    "nvidia_unified": ["ds4", "llama_cpp"],
    "other": ["llama_cpp"],
}


def _hardware_class(hw: dict[str, Any]) -> str:
    gpus = [
        n for n in hw.get("nodes", [])
        if n.get("kind") == "compute" and n.get("class") == "gpu"
    ]
    memory_classes = {
        n.get("class") for n in hw.get("nodes", []) if n.get("kind") == "memory"
    }
    vendors = {g.get("vendor") for g in gpus}
    if "apple" in vendors:
        return "apple_unified"
    if "nvidia" in vendors:
        archs = {g.get("arch") for g in gpus}
        if "gb10" in archs:
            return "nvidia_gb10"
        if "device_local" in memory_classes:
            return "nvidia_device_local"
        return "nvidia_unified"
    return "other"


def select_runtime(hw: dict[str, Any], claimed_runtimes: list[str | None]) -> str | None:
    """Pick the first preferred runtime for this hardware class that the model claims support for."""
    preferences = _RUNTIME_TABLE[_hardware_class(hw)]
    claimed = {r for r in claimed_runtimes if r}
    for runtime in preferences:
        if runtime in claimed:
            return runtime
    return None


def launch_command(
    engine: str,
    model_id: str,
    quant: str,
    context_budget: int,
    n_cpu_moe: int | None = None,
    mlock: bool = False,
    streams_l2: bool = False,
) -> str:
    """Generate the launch command for a runtime. <GGUF_PATH> is a placeholder
    for the hash-verified artifact path.

    Tiering flags are applied from the plan, not left advisory:

    - n_cpu_moe: MoE layers whose experts stay off-GPU, computed by the planner
      from the L0 tier budget and measured per-expert sizes (llama.cpp's
      `--n-cpu-moe`). None means the model fits without expert offload.
    - mlock: pin CPU-side weights in system RAM (plan's L1 tier is pinnable and
      holds expert overflow). Never combined with streams_l2.
    - streams_l2: the model overflows RAM+VRAM and streams experts from storage
      on miss; mmap must stay enabled so the page cache is the L2 tier.
    """
    if engine in ("ds4", "ds4_zgx"):
        return (
            f"ds4 serve --model <GGUF_PATH> --ctx {context_budget} "
            f"--host 127.0.0.1 --port 8080  "
            f"# {engine} for {model_id} {quant}; verify flags against ds4 docs"
        )
    if engine == "llama_cpp":
        offload = f" --n-cpu-moe {n_cpu_moe}" if n_cpu_moe else ""
        tier_flags = ""
        if mlock and not streams_l2:
            tier_flags = " --mlock"
        note = (
            "n-cpu-moe from plan L0 budget and measured expert sizes"
            if n_cpu_moe
            else "fits resident per measured sizes"
        )
        if mlock and not streams_l2:
            note += "; mlock pins L1 expert overflow in system RAM"
        if streams_l2:
            note += "; mmap default-on is the L2 stream-on-miss path"
        return (
            f"llama-server -m <GGUF_PATH> -c {context_budget} --host 127.0.0.1 --port 8080 "
            f"-ngl 999{offload}{tier_flags} --jinja  "
            f"# llama.cpp for {model_id} {quant}; {note}"
        )
    if engine == "mlx":
        return (
            f"mlx_lm.server --model <MODEL_PATH> --max-tokens 4096 --port 8080  "
            f"# MLX for {model_id} {quant}; verify against mlx-lm docs"
        )
    if engine == "vllm":
        return (
            f"vllm serve <MODEL_PATH> --max-model-len {context_budget} --port 8080  "
            f"# vLLM for {model_id} {quant}; verify against vLLM docs"
        )
    if engine == "sglang":
        return (
            f"python -m sglang.launch_server --model-path <MODEL_PATH> "
            f"--context-length {context_budget} --port 8080  "
            f"# SGLang for {model_id} {quant}; verify against SGLang docs"
        )
    return f"# no launch template for engine {engine!r} yet"
