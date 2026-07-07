# RFC 0003 — Runbooks (`runbook/v1`)

- **Status:** Draft
- **Author:** Brian Letort
- **Created:** 2026-07-06

## Summary

The question people actually have is not "what ran on someone's machine" (the compatibility matrix answers that) — it is "**what should I run on mine, and how?**" This RFC introduces the **runbook**: a distributable document, keyed to a *hardware class*, that takes an owner from diagnosis through model selection to a running, benchmarked endpoint.

A runbook is a fifth schema, `runbook/v1`, alongside the ratified four. Its defining property inherits the project's core rule:

> **Every number in a runbook is folded from committed `benchresult/v1` files or it does not appear.** Runbook prose is authored; runbook numbers never are.

Three CLI verbs make runbooks operational: `frontier runbook match` (profile this machine, return the runbooks that apply), `frontier runbook render` (YAML → readable markdown), and `frontier runbook verify` (CI gate: every cited number has a backing result file).

## Motivation

The matrix, planner, and provenance rules are the evidence engine — but the deliverable a person can *download and follow* does not exist. Three observations drive the design:

1. **Hardware clusters into classes.** A GB10, an M5 Max 128 GB, and a 16 GB-VRAM/128 GB-DDR laptop are three different answers to the same question, and thousands of people own machines in each class. Guidance written once per class serves all of them.
2. **Recommendations without provenance are content marketing.** The internet is full of "run model X on hardware Y" posts with unreproducible numbers. A runbook whose performance claims are mechanically traceable to hash-pinned, twice-reproduced result files is a different kind of artifact — and CI can enforce the difference.
3. **Community scale requires structure.** "Contribute a runbook for your hardware class" only works if a runbook is a schema-validated instance a PR bot can check, not a wiki page.

## The `runbook/v1` schema

Top-level: `schema_version`, `runbook_id`, `title`, `status`, `hardware_class`, `diagnosis`, `model_menu[]`, `expected[]`, `troubleshooting[]`, `provenance`.

### `hardware_class` — the matcher

A runbook applies to a machine when its `hwprofile/v1` resource graph satisfies a list of **node predicates** — deliberately a filter, not a model:

```yaml
hardware_class:
  name: unified_128gb
  description: Coherent unified-memory machines in the ~120-140 GB class (GB10, M5 Max 128GB)
  require:
  - kind: memory
    class: [unified]
    capacity_gb: { min: 110, max: 145 }
  - kind: compute
    class: [gpu]
  known_profiles:            # committed profiles confirmed to match (informative)
  - apple_m5_max_137gb_detected
  - gb10_128gb
```

Each predicate names a node `kind`, optionally a `class` list, and optionally `capacity_gb` / `measured` minimums (e.g. `seq_read_gbps: { min: 3.0 }`). A machine matches when every `require` entry is satisfied by at least one node. Predicates test **measured or capacity fields only** — a predicate can never reference a number the profile does not carry, and an unmeasured field fails a minimum-bound predicate (honest degradation: unmeasured machines match fewer runbooks, not more).

### `model_menu` — what to run

An ordered list of catalog models with the planner's verdict *for this class*:

```yaml
model_menu:
- modelprofile: deepseek-v4-flash/q2_imatrix
  verdict: recommended            # from a committed plan/v1 file
  plan_ref: plans/m5max_dsv4flash_q2_chat_16k.yaml
  role: daily_driver              # daily_driver | capability_ceiling | not_viable
  notes: Fits resident in unified memory with headroom for 16K context.
- modelprofile: glm-5.2/q2_routed
  verdict: not_recommended
  plan_ref: plans/gb10_glm52_q2_refusal.yaml
  role: not_viable
  notes: 238.6 GB artifact exceeds the class's unified capacity; the refusal is committed.
```

`verdict` must agree with the referenced committed plan — `runbook verify` checks it. Refusals are listed, not omitted: telling someone what *not* to download is half the product.

### `expected` — the folded numbers

```yaml
expected:
- modelprofile: deepseek-v4-flash/q2_imatrix
  workload: coding_agent
  decode_tps: 7.41
  p95_ms: 168
  context: 8192
  usability: agent_capable
  source: m5max-dsv4q2-coding-run3     # benchresult result_id — required
```

Every entry requires a `source` naming a committed `benchresult/v1` `result_id`; `runbook verify` fails if the file is missing or the numbers disagree with it. Combinations without results are either absent or listed with `unmeasured: true` and no numbers.

### `diagnosis` and `troubleshooting`

Authored prose with structure: `diagnosis` is the ordered bring-up (install, `frontier detect`, what to check in the emitted profile, which measurements matter for this class); `troubleshooting` is a list of `symptom` / `check` / `fix` entries. These sections carry no performance numbers.

### Rendering and distribution

`frontier runbook render` produces a self-contained markdown document from the YAML — the downloadable artifact. Rendered runbooks live in `runbooks/rendered/` and are regenerated by CI (like the matrix): the YAML is the source of truth, the rendering is a build product, and hand-edits to rendered output are rejected the same way hand-edits to the matrix are.

## Repository layout

```
runbooks/                  runbook/v1 YAML (source of truth)
runbooks/rendered/         generated markdown (never hand-edited)
schemas/runbook.v1.json    the JSON Schema (also bundled in src/)
```

## CLI

| Verb | Behavior |
|---|---|
| `frontier runbook match [--profile P]` | Evaluate every committed runbook's matcher against a profile (default: run detect); list matches with their model menus |
| `frontier runbook render <runbook>` | YAML → markdown |
| `frontier runbook verify [path]` | For CI: schema-validate, resolve every `expected[].source` to a committed result and re-check the numbers, re-check `model_menu[].verdict` against the referenced plans |

## What this does not do

- **No new measurement semantics.** Runbooks introduce zero new number-producing paths; they cite results the existing pipeline produced.
- **No per-machine tuning.** A runbook addresses a class. The per-machine answer remains `frontier plan` against your own detected profile; runbooks tell you it is worth running.
- **No held policies.** Runbooks reference only the open policies enumerated in plan/v1 (see [IP_NOTICE.md](../IP_NOTICE.md)).

## Alternatives considered

- **Runbooks as plain markdown guides:** zero schema work, but numbers drift from results the moment they are written, and community contributions cannot be mechanically checked. The provenance rule is the product; markdown cannot carry it.
- **Generating runbooks entirely from plans + results (no authored YAML):** attractive purity, but diagnosis steps, class descriptions, and troubleshooting are genuinely human knowledge. The split chosen — authored prose, folded numbers — keeps each kind of content where it belongs.
- **Per-machine (not per-class) runbooks:** that is what `frontier plan` already is. The class granularity is what makes a runbook shareable.
