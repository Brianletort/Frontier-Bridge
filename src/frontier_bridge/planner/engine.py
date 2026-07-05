"""Rules-based planner v0: hwprofile + modelprofile + workload -> plan/v1.

Documented heuristics, no ML. Pipeline:

    fit check -> resident placement -> expert-tier budgeting -> runtime selection
    -> risk annotation

Refusal behavior is a feature: when the numbers don't work, or we don't have
enough measured/claimed data to reason at all, the planner emits
``verdict: not_recommended`` with machine-readable reasons instead of a
hopeful plan.

Heuristics used when measured data is missing (always disclosed in the plan):

- Estimated model size from parameter count when the artifact size is unknown:
  ``size_gb ~= params_total_b * bits_per_weight / 8``, with documented
  bits-per-weight per quant family (see _QUANT_BPW). Plans built on estimates
  are capped at ``verdict: experimental``.
- Dense-resident share for MoE models when unmeasured: active-parameter bytes
  at the quant's bits-per-weight, as a floor for what must stay resident.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from frontier_bridge.adapters import launch_command, select_runtime
from frontier_bridge.catalog import get_hardware_profile, get_model_profiles
from frontier_bridge.results import fold_matrix, load_results

WORKLOADS = {
    "chat",
    "coding_agent",
    "long_context",
    "tool_calling",
    "rag",
    "batch_summarization",
    "multi_agent",
    "data_analysis",
}

# Documented bits-per-weight estimates per quant family, used ONLY when the
# artifact size is unmeasured, and always disclosed as an estimate in the plan.
_QUANT_BPW = {
    "q2": 2.8,
    "q4": 4.7,
    "q8": 8.5,
    "fp8": 8.0,
    "bf16": 16.0,
}

# Preference order when --quant is not forced: highest quality first.
_QUANT_PREFERENCE = ["q4", "fp8", "q8", "q2", "bf16"]

_HOT_KV_WINDOW_CAP = 32768


class PlanError(Exception):
    """Unrecoverable planner input error (unknown profile, workload, etc.)."""


def _quant_family(quant: str) -> str | None:
    for family in _QUANT_BPW:
        if quant.lower().startswith(family):
            return family
    return None


def _estimate_size_gb(params_total_b: float | None, quant: str) -> float | None:
    family = _quant_family(quant)
    if params_total_b is None or family is None:
        return None
    return round(params_total_b * _QUANT_BPW[family] / 8, 1)


def _memory_nodes(hw: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for node in hw.get("nodes", []):
        if node.get("kind") == "memory":
            grouped.setdefault(node.get("class", "unknown"), []).append(node)
    return grouped


def _storage_nodes(hw: dict[str, Any]) -> list[dict[str, Any]]:
    return [n for n in hw.get("nodes", []) if n.get("kind") == "storage"]


def _pick_quant(entries: list[dict[str, Any]], forced: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pick (model_profile_data, artifact) for the requested or preferred quant."""
    available: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for data in entries:
        for artifact in data.get("artifacts", []):
            available.append((artifact.get("quant", ""), data, artifact))
    if not available:
        raise PlanError("model has no artifacts in its profiles")
    if forced:
        for quant, data, artifact in available:
            if quant == forced:
                return data, artifact
        raise PlanError(
            f"quant {forced!r} not found; available: {sorted(q for q, _, _ in available)}"
        )
    for preference in _QUANT_PREFERENCE:
        for quant, data, artifact in available:
            if _quant_family(quant) == preference:
                return data, artifact
    return available[0][1], available[0][2]


def _lookup_verified_result(
    repo_root: Path, hardware_id: str, modelprofile: str, workload: str
) -> dict[str, Any] | None:
    """Best verified matrix row for this exact combination, or None."""
    rows = fold_matrix(load_results(repo_root))
    for row in rows:
        if (
            row.status == "verified"
            and row.hwprofile == hardware_id
            and row.modelprofile == modelprofile
            and row.workload == workload
        ):
            return {
                "decode_tps": row.decode_tps,
                "result_id": row.result_id,
                "usability": row.usability,
            }
    return None


def generate_plan(
    repo_root: Path,
    model_id: str,
    hardware_id: str,
    workload: str,
    context_budget: int,
    quant: str | None = None,
    engine_override: str | None = None,
) -> dict[str, Any]:
    """Generate a plan/v1 dict. Raises PlanError only for bad inputs;
    infeasible combinations return a not_recommended plan instead."""
    if workload not in WORKLOADS:
        raise PlanError(f"unknown workload {workload!r}; known: {sorted(WORKLOADS)}")
    if context_budget < 1:
        raise PlanError("context budget must be >= 1")

    hw_entry = get_hardware_profile(repo_root, hardware_id)
    if hw_entry is None:
        raise PlanError(
            f"hardware profile {hardware_id!r} not found; see `frontier catalog hardware`"
        )
    model_entries = get_model_profiles(repo_root, model_id)
    if not model_entries:
        raise PlanError(f"model {model_id!r} not found; see `frontier catalog models`")

    model, artifact = _pick_quant([e.data for e in model_entries], quant)
    chosen_quant = artifact.get("quant", "unknown")
    hw = hw_entry.data
    arch = model.get("architecture") or {}

    plan_id = f"{hardware_id}-{model_id}-{chosen_quant}-{workload}-{context_budget}".replace("_", "-")
    base: dict[str, Any] = {
        "schema_version": "plan/v1",
        "plan_id": plan_id.lower(),
        "inputs": {
            "hwprofile": hardware_id,
            "modelprofile": f"{model_id}/{chosen_quant}",
            "workload": workload,
            "context_budget": context_budget,
        },
    }

    def refuse(reasons: list[str]) -> dict[str, Any]:
        return {**base, "verdict": "not_recommended", "reasons": reasons}

    # --- Fit check ------------------------------------------------------
    reasons: list[str] = []

    if arch.get("params_total_b") is None:
        reasons.append(
            "insufficient_model_data: params_total_b is null (unverified upstream); "
            "cannot size the model — verify the model profile first"
        )

    context_max = arch.get("context_max")
    if context_max is not None and context_budget > context_max:
        reasons.append(
            f"context_budget_exceeds_claimed_max: {context_budget} > {context_max}"
        )

    memory = _memory_nodes(hw)
    primary_pool = (memory.get("device_local") or memory.get("unified") or [None])[0]
    if primary_pool is None:
        reasons.append("no_gpu_memory: hardware profile has no device_local or unified memory node")

    if reasons:
        return refuse(reasons)

    system_pool = (memory.get("system") or [None])[0]
    storage = _storage_nodes(hw)
    storage_pool = storage[0] if storage else None

    size_gb = artifact.get("size_gb")
    size_estimated = False
    if size_gb is None:
        size_gb = _estimate_size_gb(arch.get("params_total_b"), chosen_quant)
        size_estimated = size_gb is not None
    if size_gb is None:
        return refuse(
            [
                "insufficient_model_data: artifact size unmeasured and quant family "
                f"{chosen_quant!r} has no documented bits-per-weight estimate"
            ]
        )

    primary_gb = primary_pool.get("capacity_gb") or 0
    system_gb = (system_pool or {}).get("capacity_gb") or 0
    memory_total_gb = primary_gb + system_gb
    is_moe = arch.get("type") == "moe"

    # Documented headroom margins. Apple unified memory caps GPU allocations
    # near the Metal working-set limit (~75% of capacity, observed empirically
    # on the M5 Max — see known_failure_modes); discrete VRAM gets 0.9.
    primary_headroom = 0.75 if primary_pool.get("class") == "unified" else 0.9

    fits_in_memory = size_gb <= memory_total_gb * 0.9  # documented headroom margin
    can_stream = is_moe and storage_pool is not None
    if not fits_in_memory and not can_stream:
        return refuse(
            [
                f"model_does_not_fit: ~{size_gb}GB ({'estimated' if size_estimated else 'measured'}) "
                f"exceeds usable memory ~{round(memory_total_gb * 0.9, 1)}GB and "
                "no MoE+storage streaming path is available"
            ]
        )

    # --- Resident placement ----------------------------------------------
    primary_id = primary_pool["id"]
    resident = {
        "dense_core": primary_id,
        "router": primary_id,
        "shared_experts": primary_id,
    }

    # Dense-resident floor: measured from GGUF headers when available
    # (memory_model.dense_resident_gb, written by `frontier catalog
    # inspect-gguf`); falls back to active params at quant bpw (documented
    # heuristic, disclosed as a risk).
    memory_model = model.get("memory_model") or {}
    dense_floor_gb = (memory_model.get("dense_resident_gb") or {}).get(chosen_quant)
    dense_floor_measured = dense_floor_gb is not None
    if dense_floor_gb is None:
        dense_floor_gb = _estimate_size_gb(arch.get("params_active_b"), chosen_quant)
    per_expert_gb = (memory_model.get("per_expert_gb") or {}).get(chosen_quant)

    # --- Expert-tier budgeting --------------------------------------------
    l0_budget = None
    if dense_floor_gb is not None and primary_gb:
        l0_budget = max(round(primary_gb * primary_headroom - dense_floor_gb, 1), 0)

    def _expert_capacity(budget_gb: float | None) -> int | None:
        """How many (expert, layer) slices fit in a tier — measured units only."""
        if budget_gb is None or per_expert_gb is None or per_expert_gb <= 0:
            return None
        return int(budget_gb / per_expert_gb)

    routed_experts: dict[str, Any] = {
        "l0": {
            "node": primary_id,
            "budget_gb": l0_budget,
            "policy": "layer_aware_lru",
            "expert_layer_capacity": _expert_capacity(l0_budget),
        },
    }
    if system_pool is not None:
        l1_budget = round(system_gb * 0.7, 1) if system_gb else None
        routed_experts["l1"] = {
            "node": system_pool["id"],
            "budget_gb": l1_budget,
            "policy": "lru",
            "pinned": bool(system_pool.get("pinnable") is True),
            "expert_layer_capacity": _expert_capacity(l1_budget),
        }
    if storage_pool is not None:
        routed_experts["l2"] = {"node": storage_pool["id"], "mode": "stream_on_miss"}

    kv_cache: dict[str, Any] = {
        "hot": {"node": primary_id, "window_tokens": min(context_budget, _HOT_KV_WINDOW_CAP)},
    }
    if system_pool is not None:
        kv_cache["warm"] = {"node": system_pool["id"]}
    if storage_pool is not None:
        kv_cache["cold"] = {"node": storage_pool["id"], "persist": True}

    # --- Runtime selection --------------------------------------------------
    claimed_runtimes = [
        r.get("runtime")
        for r in model.get("runtime_support", [])
        if r.get("status") in ("claimed", "verified")
    ]
    if engine_override is not None:
        if engine_override not in claimed_runtimes:
            raise PlanError(
                f"engine {engine_override!r} is not in the model's claimed runtimes "
                f"({sorted(r for r in claimed_runtimes if r)})"
            )
        engine = engine_override
    else:
        engine = select_runtime(hw, claimed_runtimes)
    if engine is None:
        return refuse(
            [
                "no_supported_runtime: none of the model's claimed runtimes "
                f"({claimed_runtimes}) match this hardware class"
            ]
        )

    # --- Streaming feasibility (measured units only) --------------------------
    # Worst-case decode miss cost per token: every routed activation misses and
    # streams from storage. bytes/token = active_per_token x n_moe_layers x
    # per-(expert,layer) size. Computed only from measured values — when any
    # input is unmeasured this stays None and a risk is recorded instead.
    routed_total = (arch.get("experts") or {}).get("routed_total")
    routed_total_gb = (
        (memory_model.get("measurement") or {}).get("routed_experts_gb") or {}
    ).get(chosen_quant)
    n_moe_layers: int | None = None
    if per_expert_gb and routed_total and routed_total_gb:
        n_moe_layers = round(routed_total_gb / (per_expert_gb * routed_total))

    streaming: dict[str, Any] | None = None
    if not fits_in_memory and per_expert_gb:
        active_per_token = (arch.get("experts") or {}).get("active_per_token")
        ssd_bw = ((storage_pool or {}).get("measured") or {}).get("seq_read_gbps")
        if n_moe_layers and active_per_token:
            worst_case_gb_per_token = round(
                active_per_token * n_moe_layers * per_expert_gb, 2
            )
            streaming = {
                "worst_case_miss_gb_per_token": worst_case_gb_per_token,
                "moe_layers": n_moe_layers,
                "storage_seq_read_gbps": ssd_bw,
                "worst_case_miss_seconds_per_token": (
                    round(worst_case_gb_per_token / ssd_bw, 2) if ssd_bw else None
                ),
                "source": "computed_from_measured_gguf_headers_and_links",
            }

    # llama.cpp expert offload: how many MoE layers' experts must stay off-GPU,
    # from the measured routed size vs the L0 budget.
    n_cpu_moe: int | None = None
    if (
        n_moe_layers
        and routed_total_gb
        and l0_budget is not None
        and routed_total_gb > l0_budget
    ):
        overflow_fraction = 1 - (l0_budget / routed_total_gb)
        n_cpu_moe = min(n_moe_layers, max(1, math.ceil(n_moe_layers * overflow_fraction)))

    # --- Risk annotation -----------------------------------------------------
    risks: list[str] = []
    if not fits_in_memory:
        risks.append("decode_latency_spikes_on_expert_miss")
        ssd_bw = ((storage_pool or {}).get("measured") or {}).get("seq_read_gbps")
        if ssd_bw is None:
            risks.append("ssd_bandwidth_unmeasured_streaming_performance_unknown")
        if streaming is None:
            risks.append("streaming_cost_not_computable_missing_measured_expert_sizes")
        elif (streaming.get("worst_case_miss_seconds_per_token") or 0) > 1.0:
            risks.append("worst_case_all_miss_decode_exceeds_1s_per_token")
    if size_estimated:
        risks.append("model_size_estimated_from_param_count_not_measured")
    if dense_floor_gb is None:
        risks.append("dense_resident_footprint_unknown")
    elif not dense_floor_measured:
        risks.append("dense_resident_estimated_not_measured")
    if context_max is None:
        risks.append("context_max_unverified")
    if workload in ("coding_agent", "multi_agent", "tool_calling"):
        risks.append("agent_workloads_are_decode_latency_sensitive")

    # Anything built on estimates or streaming is experimental, never recommended.
    verdict = "experimental" if (size_estimated or not fits_in_memory) else "recommended"

    # Expected values are filled from prior verified benchmark results for the
    # same (hardware, model/quant, workload) — never hand-typed. Unmeasured
    # combinations stay null/unrated.
    expected: dict[str, Any] = {
        "decode_tps": {"p50": None, "source": None},
        "usability_class": "unrated",
    }
    prior = _lookup_verified_result(
        repo_root, hardware_id, f"{model_id}/{chosen_quant}", workload
    )
    if prior is not None:
        expected["decode_tps"] = {
            "p50": prior["decode_tps"],
            "source": prior["result_id"],
        }
        expected["usability_class"] = prior["usability"]
    if streaming is not None:
        expected["streaming"] = streaming

    return {
        **base,
        "verdict": verdict,
        "placement": {
            "resident": resident,
            "tiered": {"routed_experts": routed_experts, "kv_cache": kv_cache},
        },
        "phases": {
            "prefill": {"batch": "auto", "notes": "throughput-bound; expert misses tolerable"},
            "decode": {
                "target_p95_ms": None,
                "prefetch": "static_hotlist",
                "notes": "miss-sensitive; protect the active path",
            },
        },
        "runtime": {
            "engine": engine,
            "build": None,
            "launch": launch_command(
                engine, model_id, chosen_quant, context_budget, n_cpu_moe=n_cpu_moe
            ),
        },
        "expected": expected,
        "risks": risks,
    }
