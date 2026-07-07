# Spike Protocol — Dual-Node Thunderbolt Bridge

**Status:** protocol defined; execution needs the two Lenovo machines side by
side with a Thunderbolt cable.
**Rule:** decision by spike, not by debate. This document is the spike.
**Schema note:** how a second machine is *described* is ratified in
[RFC 0005](../rfcs/0005-fleet.md) (remote node = generalized RFC 0002 island).
This spike decides whether the topology is *usable*.

## Question

Two near-identical Linux laptops (i9, 128 GB RAM, 16 GB VRAM each — one RTX
5000 Ada, one RTX 4090) are joined host-to-host by a Thunderbolt cable using
the Linux `thunderbolt-net` IP link. Does a **pipeline/layer split** across
them run a model that fits *neither machine alone* — GLM-5.2 Q2 at 238.6 GB
against 128 GB of RAM per box — at a `usable` decode rating on real suites?

## Why it might work (and what would kill it)

- **For:** at a layer cut, the per-token cross-node traffic is the hidden-state
  activation vector at the boundary — kilobyte-scale, not weight-scale. The
  link therefore costs *latency* per token, not bandwidth. Combined, the pair
  has ~256 GB RAM + 32 GB VRAM: room for the full routed-expert pool of a
  744B-class Q2 artifact split roughly in half.
- **Against:** per-token latency is exactly what decode cannot afford
  (architecture.md's phase rule). An IP-over-Thunderbolt round trip added to
  every token, times pipeline imbalance, times RPC serialization overhead,
  may push p95 past usable. Nobody's marketing numbers answer this; the
  cable does.
- **Ruled out in advance:** tensor parallelism across the link. It moves
  activations *per layer* between devices at bandwidths external links do not
  have. The refusal math goes in the writeup regardless of outcome.

## Candidates

| # | Approach | Effort | Where it lives |
|---|---|---|---|
| A | llama.cpp RPC backend (`rpc-server` on one box, `--rpc` layer split from the other) | none (config) | existing runtime |
| B | Single-box baseline: GLM-5.2 Q2 on one laptop, NVMe tier for the overflow (~110 GB past RAM) | none (config) | existing runtime |

B is the honest comparator: the dual-node result only matters if it beats the
best *one* machine can do with its own SSD.

## Protocol (5 working days)

**Day 1 — The link, measured.** Bring up `thunderbolt-net` on both boxes.
Measure the IP link (iperf3 + latency under load) and record it as a
`hwprofile/v1` link (`via: thunderbolt`) in a combined `lenovo_pair_tb4`
profile — measured numbers or null, as always. Class expectation is
single-digit GB/s and sub-millisecond latency, but expectations are not
numbers: the profile carries what iperf3 and ping measured.

**Day 2 — RPC overhead, isolated.** DeepSeek V4 Flash Q2 (107 GB — fits one
box) run two ways: locally, and layer-split over RPC. The delta is the pure
RPC + link tax, uncontaminated by memory pressure. Record decode tps and
p50/p95/p99 for both via `frontier bench`. If the tax alone pushes a
comfortably-fitting model below `usable`, stop here — that is a valid spike
outcome, published as a refusal with numbers.

**Day 3–4 — The prize run.** GLM-5.2 Q2 split across both machines
(dense core + early layers on one, late layers on the other; VRAM for
attention/KV per side). Sweep the split point around the measured per-box
memory ceiling. Bench coding_agent and chat suites x3 runs, full pins.
Also run baseline B on the 4090 box (single machine + NVMe tier) with the
same suites.

**Day 5 — Head-to-head + writeup.** A-vs-B decision memo committed to
`docs/`. Combined profile, plans, and benchresults committed whatever the
verdict.

## Decision rule

Adopt dual-node as a **supported topology** (documented in a runbook, plans
generated against the combined profile) only if:

1. GLM-5.2 Q2 dual-node decode rates `usable` or better on the coding_agent
   suite (usability thresholds as documented in the bench harness), and
2. it beats single-box baseline B on decode tps *and* p95 at equal context.

Otherwise: publish the refusal with the measured link numbers and the
per-token tax, and keep the pair as two independent fleet machines. A
committed `not_recommended` verdict for the combined profile is a fully
successful spike outcome.

## Instrumentation requirements

- Link bandwidth and latency recorded on the combined profile (`iperf3`
  figures, tool version pinned) — the tier-normalization rule (RFC 0002)
  needs the measured wire, not the cable's marketing class.
- Per-token latency percentiles from `frontier bench` on both sides of every
  comparison; decode-only, prefill excluded.
- `risks: [dual_node_requires_runtime_rpc_support]` carried by every plan
  against the combined profile until verified rows exist (RFC 0005).

## Exit artifacts

1. `hardware_profiles/lenovo_pair_tb4.yaml` (combined profile, measured link).
2. Committed plans and benchresults for A and B, reproduced per the
   verified rule.
3. Decision memo in `docs/` — adopt as supported topology, or refusal with
   numbers.
4. If adopted: a `dual-node-256gb` runbook draft and a runtime-adapter note
   for the RPC launch invocation.
