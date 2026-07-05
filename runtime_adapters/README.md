# Runtime adapters (v0: wrappers, not integrations)

Frontier Bridge is not another inference runtime. Adapters generate the exact
launch command and configuration for an existing runtime from a `plan/v1`
document, and (in Phase 4+) capture its stdout/telemetry. Deep integrations
come later.

Adapter code lives in `src/frontier_bridge/adapters.py`. This directory
documents per-runtime status.

| Runtime | Status | Notes |
|---|---|---|
| ds4 / DwarfStar | wrapper (launch command) | Reference for SSD expert streaming on Metal; CUDA path is the Phase 5 target |
| ds4-zgx (GB10) | wrapper (launch command) | GB10/SM121 build of ds4 |
| llama.cpp | wrapper (launch command) | Broadest hardware coverage; `-ngl` / MoE CPU-offload flags map to plan tiers |
| MLX | wrapper (launch command) | Apple unified memory path |
| vLLM | wrapper (launch command) | Documented path for datacenter-class deployments |
| SGLang | wrapper (launch command) | Documented path; GLM-5.2 upstream lists support |
| TensorRT-LLM | not started | Planned, post-v0.1 |
| KTransformers | not started | Planned, post-v0.1 |

All generated commands embed `<GGUF_PATH>` / `<MODEL_PATH>` placeholders — the
operator supplies the hash-verified artifact path. Commands are printed for
review, never executed silently. Verify flags against each runtime's current
docs; runtimes evolve faster than this table.
