# RFC 0004 — Catalog Admission Policy

- **Status:** Draft
- **Author:** Brian Letort
- **Created:** 2026-07-06

## Summary

The model catalog is a **recommendation, not a directory**. This RFC defines written admission criteria, size classes with a hard cap, and a retirement policy, so "which models belong in the catalog" is a rule that PRs can be checked against — not a per-PR judgment call.

It adds one additive block to `modelprofile/v1` (allowed under the v1 freeze): an optional `catalog` object recording admission status and the claimed evidence behind it.

## Motivation

Frontier Bridge exists to answer "what is the most capable thing my hardware can actually run." That answer degrades in both directions:

- **Too many models** and the catalog becomes a directory — every quant of everything, no signal, and each entry carries maintenance cost (hash pinning, memory-model measurement, planner coverage, benchmark rows).
- **Unwritten curation** and catalog PRs become popularity contests decided by whoever reviews them, which is exactly the kind of judgment the provenance rules were written to eliminate elsewhere.

The fix is the same one used for results: write the rule down, apply it equally to maintainer, sponsored, and community submissions (per [GOVERNANCE.md](../GOVERNANCE.md)).

## Admission criteria

A model/quant enters the catalog only when **all** of the following hold:

1. **Open weights, downloadable.** Publicly downloadable artifacts; no application gates that block reproduction.
2. **Enterprise-usable license.** `license.commercial_ok: true` in the profile, with the license named. `unknown` is not admissible.
3. **Pinnable artifacts.** GGUF (or other supported format) artifacts with per-shard sha256 available — a catalog entry that cannot back a verified row is dead weight.
4. **Frontier-class capability, claimed with sources.** The model must be at or near the top of its size class on published evaluations, recorded as `claimed` evidence with URLs in the `catalog.evidence` block. The bar is deliberately relative ("top of its size class at admission time"), not an absolute score — absolute thresholds rot as benchmarks saturate.
5. **Agent capability.** Tool-calling/agent use must be claimed by upstream (and is expected to be exercised by the `coding_agent` and `tool_calling` bench suites). Chat-only models are out of scope regardless of quality.
6. **Placement-relevant architecture.** MoE sparsity or another property that makes hierarchy planning meaningful. Dense models are admissible only when they are the capability ceiling of a class the hardware targets (e.g. a dense ~120B that fits a 128 GB unified machine).

## Size classes and the cap

Classes are by **total parameters** (what storage and tiers must hold), not active:

| Class | Range | Cap |
|---|---|---|
| `compact` | ≤ 150B total | 2 models |
| `mid` | 150–400B total | 2 models |
| `frontier` | > 400B total | 3 models |

**The cap counts models, not quants** — a model's Q2 and Q4 artifacts are one entry. When an admission would exceed the cap, something must retire in the same PR. The caps are policy, revisable by amending this RFC — never silently.

## Status lifecycle

```
provisional → admitted → retired
```

- **provisional** — meets criteria 1–3 and 5–6; evidence block incomplete (missing citations). Planner and bench work may proceed; the model is not shown as a recommendation in runbooks.
- **admitted** — all six criteria met, evidence cited. Eligible for runbook model menus.
- **retired** — superseded within its class or evidence invalidated. Profile and matrix rows are **kept** (history and reproducibility); the model no longer appears in runbook menus and its `catalog.status` says why.

## Schema change (additive)

`modelprofile/v1` gains an optional `catalog` block:

```yaml
catalog:
  status: admitted            # provisional | admitted | retired
  size_class: mid             # compact | mid | frontier
  since: 2026-07-06
  evidence:                   # claimed, with sources — required for admitted
  - claim: top-2 open-weights coding-eval placement in class at admission
    source: https://example.com/eval-source
  retired_reason: null        # required when status is retired
```

Profiles without a `catalog` block remain valid (they are simply uncatalogued — profile format is usable by anyone for anything; the catalog is what *this project* recommends and maintains).

## Audit of the current seven families

License claims were sourced during this audit (each profile now carries `license.source`); criterion 4 (eval evidence with citations) remains fieldwork before any entry moves past `provisional`:

| Model | Total params (claimed) | Class | License (sourced) | Assessment |
|---|---|---|---|---|
| glm-5.2 | 744 | frontier | MIT, `commercial_ok: true` | `provisional` — v0.1 primary target; eval evidence block needed |
| deepseek-v4-flash | 284 | mid | MIT, `commercial_ok: true` | `provisional` — v0.1 primary target; the only model with verified rows |
| qwen3-coder-480b | 480.2 | frontier | Apache 2.0, `commercial_ok: true` | `provisional` — eval evidence block needed |
| kimi-k2.6 | 1026.4 | frontier | Modified MIT, `commercial_ok: true` (attribution above 100M MAU / $20M-month revenue) | Contends with k2.7-code within family — one retires |
| kimi-k2.7-code | 1026.4 | frontier | Modified MIT, `commercial_ok: true` (same attribution clause) | `provisional` — upstream positions it as the K2.6 successor for coding; if admitted, kimi-k2.6 retires |
| minimax-m3 | 426.0 | frontier | MiniMax Community License, `commercial_ok: false` (commercial use requires prior written authorization) | **Inadmissible** under criterion 2 — retire from the catalog; profile and any rows are kept per the no-deletion rule |
| gpt-oss-120b | 116.8 | compact | Apache 2.0, `commercial_ok: true` | `provisional` — criterion 6 via the dense-ceiling clause |

Consequences, resolved by sourcing rather than debate:

1. **minimax-m3 is out** — the license fails the enterprise-usable criterion on its face. This is the policy working: "open weights" and "enterprise-usable" are different claims, and only the sourced license text distinguishes them.
2. **The `frontier` class lands at exactly the cap** — glm-5.2, qwen3-coder-480b, and one Kimi (k2.7-code presumptively, as upstream's designated successor to k2.6 for coding workloads; confirmed or reversed by the eval-evidence pass).
3. **The `mid` class has one occupant** (deepseek-v4-flash) and an open seat.

## What this does not do

- **No quality verification.** Admission evidence is `claimed` (upstream evals, with sources). Verifying *usability on specific hardware* remains the benchmark pipeline's job; the catalog never asserts it.
- **No deletion.** Retirement removes recommendation status, never data. Verified rows for retired models stay in the matrix.
- **No gatekeeping of profiles.** Anyone can maintain out-of-catalog profiles in a fork or PR them as examples; the catalog governs what ships in `model_profiles/` with maintenance commitment and runbook placement.

## Alternatives considered

- **Absolute eval-score thresholds** (e.g. "≥ X on benchmark Y"): objective-looking but rots quickly — benchmarks saturate, get contaminated, and fall out of use. Relative-to-class placement with cited sources ages better and is equally auditable.
- **No cap, criteria only:** criteria alone admit everything good, and the catalog drifts toward a directory. The cap forces the ranking conversation to happen in the open, in PRs, with sources.
- **A separate catalog file** instead of a per-profile block: one more thing to drift from the profiles it describes. The profile is where the license, artifacts, and memory model already live; admission status belongs next to them.
