# FAQ

Honest answers to the questions we expect — from the LocalLLaMA crowd, from enterprise architects, and from skeptics of both. Where an answer depends on unmeasured facts, we say so and point at the roadmap item that will produce the data.

## Why not just use llama.cpp directly?

You should — Frontier Bridge launches it for you. llama.cpp, ds4, MLX, vLLM, and SGLang are excellent runtimes, and this project deliberately is not one. What none of them owns is the layer above: given *this* machine and *this* model, what placement is even sensible, which runtime and flags express it, and what did it actually measure? Today, answering that means archaeology across Reddit threads and vendor claims. Frontier Bridge turns it into `detect → plan → run → bench`, with every number carrying provenance. If you already know your `-ngl` and `--n-cpu-moe` values for your exact machine and model, you are ahead of the tool — contribute a benchmark result so everyone else is too.

## Won't SSD streaming destroy my drive?

The honest answer: streaming is read-heavy, and NAND endurance is rated in *writes*, so the wear story is much better than intuition suggests — but "much better" is not a number, and we refuse to hand-wave it. Every `frontier bench` run records `ssd_total_read_gb` — a system-wide disk-read delta over the run (`null` on platforms where it can't be collected), not perfectly isolated to the model — so the answer accrues from real data as rows land: roughly how many GB a benchmark session reads at a given cache budget. Sustained-read thermal throttling on consumer NVMe is the more realistic concern, which is why the disk microbench records the tool and queue depth used.

## Is Q2 quantization even usable, or is this a party trick?

Unknown, and the model profiles say so: GLM-5.2's Q2 profile carries `known_failure_modes: q2_quality_degradation_unquantified`. The benchmark suites include task pass/fail blocks (coding agent, tool calling) precisely so usability ratings reflect *task success*, not just tokens per second. A configuration that decodes fast but fails the agent tasks gets rated accordingly. Until those rows exist, treat Q2 quality as an open question — we do.

## Why does the planner refuse instead of warning?

Because a warning next to a hopeful plan gets ignored, and the numbers this tool deals in are too expensive to ignore — a wrong "should work" costs a ~239 GB download and an evening. When parameter counts are unverified, context budgets exceed claimed maxima, or no claimed runtime matches the hardware, the planner emits `verdict: not_recommended` with machine-readable reasons. A planner you can trust when it says *yes* is one that says *no* out loud.

## What is ds4?

An SSD-expert-streaming inference runtime with a reference implementation on Apple Metal ([adapter status](../runtime_adapters/README.md)) — its cache-miss path (SSD → staging → compute-local cache) is the closest existing design to what the [CUDA streaming spike](spike_cuda_expert_streaming.md) evaluates for the RTX 6000 class. The planner selects it by default on Apple Silicon when the model claims support; pass `--engine llama_cpp` if that is what you run. Adapters are launch-command wrappers in v0.1, not deep integrations.

## What do `recommended` / `experimental` / `not_recommended` actually mean?

- `recommended` — measured artifact size fits in memory with a documented 10% headroom margin. A statement about *memory fit*, not usability: usability classes (`runs → usable → interactive → agent_capable`) are only ever assigned from benchmark results.
- `experimental` — the plan depends on estimates (size derived from parameter count) or requires storage streaming. It may work well; nothing measured says so yet.
- `not_recommended` — a refusal, with reasons. Also a first-class result in the compatibility matrix: knowing a combination does not work is knowledge worth publishing.

## Why start with GLM-5.2 and DeepSeek V4 Flash?

They bracket the interesting range. DeepSeek V4 Flash Q2 (107 GB measured) *fits resident* on a 128 GB-class machine — the clean baseline where the memory hierarchy barely matters. GLM-5.2 Q2 (238.6 GB measured) *cannot fit* on any v0.1 reference machine — the stress case where tiering is the whole story. Both are MoE with pinned GGUF artifacts and llama.cpp support. Once the format is proven, additional families (Qwen, MiniMax, Kimi) are new profiles, not new code.

## When RTX 5090? Strix Halo? My machine?

When someone with the hardware runs `frontier detect` and opens a PR — both are open gaps in the [compatibility matrix](../README.md#compatibility-matrix), and hardware profiles from machines we don't have are the single most valuable early contribution ([CONTRIBUTING.md](../CONTRIBUTING.md)). The detect path currently measures macOS/Apple Silicon and Linux/NVIDIA (incl. WSL2); other platforms start from the [manual template](../hardware_profiles/templates/manual_template.yaml).

## Why is a data center company sponsoring a workstation tool?

Because the same schema that describes a workstation describes a rack node, and Digital Realty operates at the infrastructure end of that continuum. The sponsorship funds development and reference hardware; it does not change how technical decisions are made, and the [results-integrity rule](../GOVERNANCE.md#results-integrity) applies identically to sponsored and community contributions. The project is Apache 2.0 — if the sponsor vanished tomorrow, everything here still works.

## Can I trust the numbers in the compatibility matrix?

The tool-enforced part is mechanical: `frontier bench` will not label a result `verified` unless all four pins are non-null (plan hash, model sha256, runtime commit, hwprofile) and at least two reproductions are listed — including for maintainers. That reproductions come from fresh runtime starts is process, enforced in review per the [benchmark playbook](benchmark_playbook.md), not by the tool. Anything not yet measured says `unrated (target)`, and the matrix currently says exactly that everywhere, because v0.1-dev has zero verified rows. That zero is displayed rather than hidden; it is the standard the first rows have to meet.

## How do I contribute a benchmark result?

Follow the [benchmark playbook](benchmark_playbook.md): plan, run with a hash-verified artifact, bench with the runtime commit pinned, reproduce twice, and open a PR moving the result JSON into `results/community/`. The PR template walks the pins. Results that don't meet the bar are still welcome as `claimed` — they seed the map for someone to verify.
