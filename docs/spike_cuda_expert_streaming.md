# Spike Protocol — CUDA SSD Expert Streaming (Phase E)

**Status:** protocol defined; execution starts post-launch on the RTX 6000 (WSL2).
**Rule:** decision by two-week spike, not by debate. This document is the spike.
**IP note:** this spike evaluates only open policies (mmap/page cache, explicit
LRU-class caches, static hotlists). Router-aware prefetch and adaptive session
caching are out of scope pending IP review (Gate G2) — see [IP_NOTICE.md](../IP_NOTICE.md).

## Question

For MoE decode on a 96 GB VRAM / 64 GB RAM / NVMe workstation, does an
**explicit expert cache** (NVMe → pinned-RAM staging → VRAM expert cache,
app-managed eviction) beat the **implicit path** (llama.cpp-style mmap + OS
page cache + CPU-resident experts) by enough to justify owning a runtime-level
component?

## Why it matters

- The plan's measured worst case is brutal without caching: GLM-5.2 Q2 all-miss
  decode moves ~7 GB/token (8 experts x 79 MoE layers x 11 MB), which even a
  7 GB/s NVMe serves at ~1 token/s. Locality (hot experts) is the entire game.
- The RTX 6000 class is powerful enough to matter and constrained enough to
  need this. GB10 second; GDS is unavailable under WSL2, so the pinned-RAM
  bounce path is the design there (already expressed in our link graph).

## Candidates

| # | Approach | Effort | Where it lives |
|---|---|---|---|
| A | Baseline: llama.cpp `--n-cpu-moe` + mmap paging | none (config) | existing runtime |
| B | Extend ds4's CUDA path (their Metal SSD streaming is the reference design) | medium | fork/PR to ds4 |
| C | Standalone expert-cache layer under a llama.cpp-style runtime (custom tensor-backend hook) | high | new component |

## Protocol (10 working days)

**Days 1–2 — Baseline A, measured.** GLM-5.2 Q2 on the RTX box (WSL2, ext4):
sweep `--n-cpu-moe` and VRAM expert residency; record decode tps, p50/p95/p99,
`ssd_total_read_gb` per 1K tokens, RAM/VRAM peaks via `frontier bench`. This
baseline is publishable regardless of the spike outcome.

**Days 3–4 — Expert-access traces.** Instrument router outputs (llama.cpp debug
hook or ds4 logging) across the coding_agent and long_context suites. Compute
expert reuse distance and hit rates for LRU / LFU / static-hotlist at L0 sizes
{20, 40, 60} GB. This tells us the *ceiling* an explicit cache could reach —
if simulated hit rates don't beat page-cache behavior materially, stop here
and invest elsewhere (that is a valid spike outcome).

**Days 5–8 — Prototype the winner's smallest slice.** Either B (ds4 CUDA:
port their Metal cache-miss path to cudaMemcpyAsync from pinned staging) or C
(intercept expert tensor loads under llama.cpp). One model, one quant, decode
only, correctness checked by logit comparison against baseline.

**Days 9–10 — Head-to-head + writeup.** Same plan file, same bench suites,
A vs prototype. Decision memo with numbers committed to `docs/`.

## Decision rule

Adopt the explicit-cache path only if the prototype shows **≥1.5x decode tps
or ≥2x p95 improvement** over tuned baseline A at equal VRAM budget, on real
suites (not synthetic all-miss). Otherwise: ship recipes on top of A, publish
the trace analysis, and revisit when hardware or runtimes change the math.

## Instrumentation requirements (build once, keep)

- SSD total-bytes-read per run (already in `frontier bench`) — the endurance
  question needs data before the community asks it.
- Expert cache hit rate exposed per run (`expert_cache_hit_rate` field is
  already in `benchresult/v1`, null until a runtime exposes it).
- Decode-phase-only latency percentiles (exclude prefill contamination).

## Exit artifacts

1. Decision memo with A-vs-prototype numbers (committed).
2. Reusable expert-trace tooling.
3. Either: a ds4 upstream PR / component design doc (IP-cleared parts only),
   or: documented tuned-baseline recipes per hardware profile.
