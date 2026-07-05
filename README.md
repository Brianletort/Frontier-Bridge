# Frontier Bridge

**The open bridge from workstation AI to enterprise AI infrastructure.**

Sponsored by [Digital Realty](https://www.digitalrealty.com/) · Apache 2.0 · Python 3.10+

Frontier-scale open models — GLM-5.2 at 744B total / 40B active, DeepSeek V4 class — no longer require a hyperscale cloud. They require the right use of the memory hierarchy you already own: VRAM, unified memory, system RAM, and fast NVMe, planned together. Frontier Bridge profiles your hardware, plans that hierarchy, selects a runtime, and produces reproducible benchmarks — and gives you an honest refusal when the numbers don't work.

> Stop guessing. Profile your hardware. Pick a model. Generate an execution plan. Run it. Benchmark it. Share your result.

## The problem

AI infrastructure is becoming heterogeneous — hyperscale clouds, enterprise data centers, colocation, edge, and increasingly powerful workstations. The ecosystem has strong pieces at every layer, but no one owns the bridge between them:

| Layer | Exists today | Missing |
|---|---|---|
| Models | GLM-5.2, DeepSeek V4, Qwen, MiniMax, Kimi | Knowing what you can *practically* run |
| Runtimes | llama.cpp, ds4, MLX, vLLM, SGLang, TensorRT-LLM | A unified profile-to-plan layer above them |
| Hardware | RTX 6000, RTX 5090, GB10, M-series Macs | Machine-readable, *measured* hardware profiles |
| Memory | VRAM, unified memory, RAM, NVMe | A practical memory-hierarchy planner for giant MoE |
| Trust | Blog posts, Reddit anecdotes, vendor claims | Reproducible, hash-pinned benchmark numbers |

Sparse MoE models are the unlock. A 744B-total model activates ~40B parameters per token — so the real question is not "does 744B fit in memory" but "**can the active path stay fast while inactive experts tier across RAM and SSD?**" That is a planning problem, and it's the problem this project solves.

## See it work

Everything below is real output from this repo, on real hardware — nothing staged.

**1. Profile the machine.** `frontier detect` measures what it can and refuses to guess the rest:

```yaml
profile_id: apple_m5_max_137gb_detected
nodes:
- id: unified0
  kind: memory
  class: unified
  capacity_gb: 137.4
- id: ssd0
  kind: storage
  class: internal_ssd
  measured:
    seq_read_gbps: 19.38          # measured, not quoted from a spec sheet
    bench_tool: frontier-detect python bounded read (512MB, uncached, qd1)
```

**2. Plan the model.** `frontier plan glm-5.2 --hardware apple_m5_max_137gb_detected --workload coding_agent --ctx 32768` emits a tiered execution plan:

```yaml
verdict: experimental
placement:
  resident: { dense_core: unified0, router: unified0, shared_experts: unified0 }
  tiered:
    routed_experts:
      l0: { node: unified0, budget_gb: 91.7, policy: layer_aware_lru }
      l2: { node: ssd0, mode: stream_on_miss }
runtime:
  engine: ds4
  launch: ds4 serve --model <GGUF_PATH> --ctx 32768 --host 127.0.0.1 --port 8080
risks:
- decode_latency_spikes_on_expert_miss
- model_size_estimated_from_param_count_not_measured
```

**3. Get refused when it doesn't work.** DeepSeek V4 Flash's parameter counts are unverified upstream, so the planner will not pretend:

```yaml
verdict: not_recommended
reasons:
- "insufficient_model_data: params_total_b is null (unverified upstream);
   cannot size the model — verify the model profile first"
```

That refusal is the product working as designed. A planner you can trust when it says *yes* is one that says *no* out loud.

## What Frontier Bridge is not

**Not another inference runtime.** llama.cpp, ds4, MLX, vLLM, SGLang, and TensorRT-LLM are excellent at what they do. Frontier Bridge is the layer above and beside them: hardware profiles, model profiles, memory planning, runtime selection, and a common benchmark standard. Plans emit exact launch commands for existing runtimes; they never replace them.

## Honest limits

- We do not make any frontier model run on consumer hardware. We make the best possible use of available infrastructure and turn many models from impossible into runnable, and from runnable into useful.
- Every number in this repo is either **measured** (with provenance) or **null**. We never guess.
- Facts are labeled `claimed` (upstream docs say so) or `verified` (we ran it, hash-pinned, reproduced twice). Nothing enters the compatibility matrix as verified without a backing benchmark result.

## Status

**Working today (v0.1-dev):**

| Capability | State |
|---|---|
| Four versioned schemas (`hwprofile`, `modelprofile`, `plan`, `benchresult` — [RFC 0001](rfcs/0001-resource-graph-schemas.md)) | Ratified, frozen at v1 |
| `frontier detect` | Live on macOS/Apple Silicon; Linux/NVIDIA path written and fixture-tested, pending real hardware |
| `frontier plan` (rules-based planner v0) | Working: fit check, tiered placement, runtime selection, graceful refusal |
| `frontier validate` / `frontier catalog` | Working across all committed profiles |
| Runtime adapters (ds4, ds4-zgx, llama.cpp, MLX, vLLM, SGLang) | Launch-command wrappers ([status](runtime_adapters/README.md)) |
| Test suite | 29 tests, all passing |

**Next:** benchmark harness (`frontier bench` → `benchresult/v1`), pinned GGUF hashes, first verified compatibility-matrix rows, then CUDA SSD expert streaming (NVMe → pinned RAM → VRAM cache) targeting the RTX 6000 96 GB class. Enterprise-bridge deployment profiles follow. Launch gates are defined in [docs/launch_checklist.md](docs/launch_checklist.md).

## Target hardware and models (v0.1)

| System | Memory model |
|---|---|
| RTX 6000 / RTX PRO 6000 96 GB + 64–192 GB RAM | VRAM + pinned RAM + NVMe streaming |
| DGX Spark / GB10 128 GB | Unified memory + NVMe |
| M5 Max / Mac Studio 128 GB+ | Apple unified memory + internal SSD |
| RTX 5090 32 GB + 128 GB RAM | Hybrid VRAM/RAM |

Models: **GLM-5.2** (Q2/Q4 routed GGUF) and **DeepSeek V4 Flash** (Q2/Q4 imatrix GGUF) first; future Qwen / MiniMax / Kimi / GLM / DeepSeek MoE models follow the same profile format.

## Quickstart

```bash
git clone https://github.com/Brianletort/Frontier-Bridge.git
cd Frontier-Bridge
pip install -e ".[dev]"

frontier detect                      # profile this machine → hwprofile/v1
frontier catalog hardware            # list committed hardware profiles
frontier catalog models              # list committed model profiles
frontier plan glm-5.2 --hardware m5_max_128gb --workload coding_agent --ctx 32768
frontier validate .                  # validate all committed YAML against the v1 schemas
```

## How it works

```text
Hardware Profiler  →  Model Planner  →  Execution Planner
       │                   │                   │
       ▼                   ▼                   ▼
 resource graph      behavioral traits    tiered placement
 (nodes + links,     (sparsity, routing,  (prefill/decode
  measured)           KV scaling)          phase-separated)
       │                   │                   │
       └─────────────┬─────┴───────┬───────────┘
                     ▼             ▼
              Runtime Adapter   Benchmark
              (launch command)  (benchresult/v1)
```

The core design decision: **everything is a graph of resources and links, measured not assumed.** A hardware profile has no hardcoded `vram/ram/ssd` fields — it has memory, compute, storage, and network *nodes* joined by *links* with measured bandwidth. An RTX 6000 box, a GB10 with coherent unified memory, and a Mac Studio are the same schema with different topology. So are a multi-GPU rack node and a multi-site deployment — which is what makes the enterprise bridge a schema change, not a rewrite. Details in [RFC 0001](rfcs/0001-resource-graph-schemas.md).

Prefill and decode are planned separately from day one: prefill is throughput-bound and tolerates expert-cache misses; decode is miss-sensitive and gets latency targets and prefetch policy of its own. That distinction — protecting the interactive path — is where usefulness lives.

## Compatibility matrix

Usability ratings, in order: `unrated → runs → usable → interactive → agent_capable`. A sixth label, `not_recommended`, is a first-class result.

| Hardware | Model | Quant | Mode | Context | Status |
|---|---|---|---|---|---|
| RTX 6000 96 GB / 64 GB RAM | GLM-5.2 | Q2 routed | VRAM+RAM+SSD | 32K–128K | unrated (target) |
| GB10 128 GB | GLM-5.2 | Q2/Q4 routed | unified+SSD | 32K–128K | unrated (target) |
| M5 Max 128 GB | GLM-5.2 | Q4 GGUF | Metal+SSD | 32K | unrated (target) |
| M5 Max 128 GB | DeepSeek V4 Flash | Q2 | resident/SSD | 100K+ | unrated (target) |

No row moves past `unrated` until a hash-pinned `benchresult/v1` file backs it, reproduced twice. That single rule is the brand.

## The enterprise bridge

AI will not live only in hyperscale clouds. It will run across laptops, workstations, enterprise GPU rooms, colocation facilities, private cloud, service-provider platforms, and hyperscale data centers. Frontier Bridge gives you a path upward across that continuum:

```text
workstation → lab server → private AI rack → colocation / connected AI infrastructure → cloud & partner ecosystem
```

The same profiles, plans, and benchmarks that describe a single workstation describe a rack node. When your workload outgrows the desk, [Digital Realty](https://www.digitalrealty.com/)'s platform — data centers, interconnection, cloud on-ramps, and high-density AI deployments across 300+ facilities in 55+ metros — is the natural landing zone, and this project exists to make that transition a schema change, not a rewrite. See [docs/enterprise_bridge.md](docs/enterprise_bridge.md).

## Sponsorship

Frontier Bridge is sponsored by **Digital Realty**, which sits at the center of the heterogeneous AI infrastructure ecosystem this project serves. Digital Realty is not trying to replace model providers or inference runtimes — this project is its open contribution to the missing infrastructure layer: hardware profiles, memory planning, runtime selection, benchmarks, and deployment recipes that help people get the most from the infrastructure they already have. Sponsorship does not change how technical decisions are made; see [NOTICE](NOTICE) and [GOVERNANCE.md](GOVERNANCE.md).

## Repository layout

```text
schemas/              JSON Schema files for the four v1 schemas
rfcs/                 design RFCs (schemas are ratified here)
hardware_profiles/    committed hwprofile/v1 YAML (detected or manual provenance)
model_profiles/       committed modelprofile/v1 YAML per model/quant
plans/                example plan/v1 outputs
runtime_adapters/     per-runtime adapter status
docs/                 enterprise bridge, launch checklist
src/frontier_bridge/  the CLI and library
tests/                pytest suite (29 tests)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The most valuable early contributions are hardware profiles from machines we don't have and benchmark results that reproduce.

## License

Apache 2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and [IP_NOTICE.md](IP_NOTICE.md).
