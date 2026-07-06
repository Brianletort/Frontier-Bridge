# Mission

**Frontier Bridge makes the most capable open AI models practical across heterogeneous infrastructure — from high-end workstations to enterprise-grade AI platforms.**

## The gap we fill

The ecosystem has strong pieces, but no one owns the bridge between them:

| Layer | Exists today | Gap |
|---|---|---|
| Models | GLM-5.2, DeepSeek V4, Qwen, MiniMax, Kimi | Users do not know what they can practically run |
| Runtimes | llama.cpp, MLX, ds4, vLLM, SGLang, TensorRT-LLM | Each has different assumptions and tuning knobs |
| Hardware | RTX 6000, RTX 5090, GB10, M5 Max, Mac Studio | No unified profile-to-plan system |
| Memory | VRAM, unified memory, RAM, SSD | No practical memory-hierarchy planner for giant MoE |
| Agents | Cursor, Codex, Hermes, OpenWebUI, OpenHands, Aider | No workload-aware optimization layer |
| Enterprise bridge | Colocation, CSPs, NSPs, private AI platforms | No clean path from workstation experimentation to enterprise deployment |

The gap is not "can someone hack GLM-5.2 into running once." The gap is **a platform that turns hardware into an AI execution environment.**

## The core idea

Treat AI infrastructure as a memory hierarchy. Instead of "does the model fit — yes/no," the questions are:

- What must stay in VRAM?
- What can live in unified memory?
- What can live in system RAM?
- What can stream from SSD?
- What should be prefetched, compressed, or distributed?

Sparse MoE models make this tractable: for a 744B-total / 40B-active model, the practical question is whether the active path stays fast while inactive experts are intelligently tiered.

## The infrastructure continuum

```text
High-end consumer / workstation
  → single-user local AI
  → team workstation / lab server
  → department GPU server
  → enterprise private AI rack
  → colocation / connected AI infrastructure
  → cloud, NSP, CSP, partner ecosystem
```

Frontier Bridge gives users a path upward. The same profiles, plans, and benchmarks that describe a single M5 Max describe a multi-GPU rack node — the schema is a resource graph, and bigger machines are just more nodes and links.

The continuum is the "base" in a simple operating thesis: **build the base, rent the spike.** Own the infrastructure for steady-state AI workloads at whatever rung fits, and rent cloud capacity for the peaks. Frontier Bridge is the connective tissue that makes the owned side of that equation credible — measured proof of what each rung can actually run. The upward path is sketched in [docs/enterprise_bridge.md](docs/enterprise_bridge.md).

## Principles

1. **Credibility before code.** The schemas, honest limits, and refusal behavior matter more than early features.
2. **Measurement before optimization.** Nothing is labeled verified without pinned hashes and two reproductions.
3. **Open standard before novel kernels.** The schemas and benchmark format are the durable contribution; runtimes will keep improving underneath.
4. **Not another runtime.** We generate exact launch commands for existing engines and measure the result.
