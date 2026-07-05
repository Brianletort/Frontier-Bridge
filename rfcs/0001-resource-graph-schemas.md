# RFC 0001 — Resource Graph Schemas (v1)

- **Status:** Ratified
- **Author:** Brian Letort
- **Created:** 2026-07-04

## Summary

Four independently versioned schemas form the data contract of Frontier Bridge:

| Schema | Purpose |
|---|---|
| `hwprofile/v1` | What a machine is: a graph of resources and links, measured not assumed |
| `modelprofile/v1` | What a model needs: behavioral traits, not just sizes |
| `plan/v1` | How to run a model on a machine: phase-separated, tiered placement |
| `benchresult/v1` | What actually happened: pinned, reproducible measurements |

JSON Schema validation files live in `schemas/`. Committed YAML/JSON instances are validated by `frontier validate`.

## Design principles

1. **Graph, not fields.** No hardcoded `vram/ram/ssd` fields. A hardware profile is **nodes** (compute, memory, storage, network) and **links** (interconnects with measured bandwidth/latency). Future hardware — coherent-memory stations, CXL pools, Gen6 NVMe, multi-node Ethernet — becomes new nodes/links, not a schema break.
2. **Measured wins; never guess.** Rated and measured values are recorded separately. Unknown values are `null` or the string `unknown` — never invented. Capability presence is tri-state: `true | false | unknown`.
3. **Claimed vs verified.** Any fact taken from upstream docs is `claimed`. It becomes `verified` only when a hash-pinned benchmark run backs it, reproduced twice. This distinction is the trust mechanism of the whole project.
4. **Phase separation.** Prefill and decode get independent placement and policy. Decode is miss-sensitive; prefill is throughput-bound. The schema encodes that physics from day one.
5. **Provenance everywhere.** Every instance carries `schema_version` and provenance/timestamps so results stay reproducible as hardware and models evolve.

## 1. `hwprofile/v1`

Top-level: `schema_version`, `profile_id`, `provenance`, `nodes[]`, `links[]`, optional `envelope`.

- **Nodes** have `id`, `kind` (`compute | memory | storage | network`), and kind-specific fields:
  - compute: `class` (`gpu | cpu`), `vendor`, `arch` (free-form, mapped in a maintained arch table), detected `api` block, `rated` block (nullable).
  - memory: `class` (`device_local | unified | system | cxl | remote`), `capacity_gb`, `bandwidth_gbps: { rated, measured }`, optional `attached_to`, `numa`, `pinnable`.
  - storage: `class` (`nvme | sata | internal_ssd | other`), `capacity_gb`, optional `pcie`, `measured` microbench block including the tool used.
  - network: `class`, `rated_gbps`.
- **Links** are the part most schemas omit — and the part that matters. Each has `from`, `to`, `via` (`pcie | gds | unified | ethernet | other`), and a `measured` block (e.g. `h2d_gbps`, `d2h_gbps`, `seq_read_gbps`). Optional `available: true|false|unknown` for paths like GPUDirect Storage.
- **Why links matter:** on an RTX 6000 box the planner's real constraint is not "does it fit in 96 GB" — it is the measured NVMe→RAM→VRAM chain versus decode-time expert-miss latency. Unified-memory machines (GB10, M5 Max) express as a single `unified` memory node with a different link topology. Same schema, very different machines.

## 2. `modelprofile/v1`

Top-level: `schema_version`, `model_id`, `family`, `license`, `architecture`, `artifacts[]`, `runtime_support[]`, `memory_model`, `known_failure_modes[]`.

- `architecture` captures behavioral traits the planner schedules on: `type` (`moe | dense`), total/active params, expert counts, attention/KV scaling, max context (treated as a claim until benchmarked), speculative-decode support.
- `artifacts` are hash-pinned: `format`, `quant`, `size_gb` (nullable), `source`, `sha256` (nullable until pinned).
- `runtime_support` entries carry `status: claimed | verified`.
- `memory_model` holds measured per-quant numbers the planner consumes: `dense_resident_gb`, `per_expert_gb`, `kv_per_1k_tokens_mb` — all nullable until measured.

## 3. `plan/v1`

Top-level: `schema_version`, `plan_id`, `inputs`, `placement`, `phases`, `runtime`, `expected`, `risks[]`, optional `verdict`.

- `inputs` pins the hwprofile, modelprofile/quant, workload, and context budget the plan was generated from.
- `placement.resident` maps tensor classes (dense core, router, shared experts) to memory node IDs from the hwprofile.
- `placement.tiered.routed_experts` defines up to three tiers (`l0`/`l1`/`l2`) with node, budget, and policy. **Only open policies are valid in v1:** `layer_aware_lru`, `lru`, `lfu`, `static_hotlist`, `stream_on_miss`, `none` (see [IP_NOTICE.md](../IP_NOTICE.md)).
- `placement.tiered.kv_cache` defines `hot`/`warm`/`cold` placement.
- `phases.prefill` and `phases.decode` carry independent policy and latency targets.
- `expected` values are filled from prior benchmark results with a `source` reference — never hand-typed.
- `usability_class`: `unrated → runs → usable → interactive → agent_capable`, plus `not_recommended`. The planner refusing is a feature; a refusal plan carries `verdict: not_recommended` with machine-readable `reasons`.

## 4. `benchresult/v1`

One JSON file per run. Top-level: `schema_version`, `result_id`, `pins`, `metrics`, `workload_tasks`, `environment`, `reproductions`.

- `pins`: plan hash, model artifact sha256, runtime commit, hwprofile ID. Without all pins the result cannot be labeled verified.
- `metrics`: TTFT, prefill tps, decode tps, p50/p95/p99 token latency, peak VRAM/RAM, SSD read GB/s and **total GB read** (endurance tracking), expert cache hit rate (nullable if the runtime does not expose it), context length achieved.
- `workload_tasks`: pass/fail block per task for agent-usefulness rating.
- `reproductions`: list of prior `result_id`s this run reproduces. Two reproductions are required for verified status.

The community leaderboard is a fold over these files — no separate database.

## Versioning and stability

- Each schema is versioned independently (`hwprofile/v1` may reach v2 before `plan/v1` does).
- v1 schemas are **frozen**: additive, backward-compatible changes only. Breaking changes require a new RFC.
- Validators reject unknown `schema_version` values rather than guessing.

## Alternatives considered

- **Flat fields (`vram_gb`, `ram_gb`, `ssd_gbps`):** simpler, but breaks on unified memory, CXL, and multi-node — exactly the hardware this project targets next.
- **Reusing an existing hardware description format (hwloc XML, etc.):** rich on topology, silent on measured bandwidth, model behavior, and plan/benchmark semantics. We record the measuring tool instead and keep the schema purpose-built.
