# RFC 0005 — Fleet Registry and Remote Nodes (`fleet/v1`)

- **Status:** Draft
- **Author:** Brian Letort
- **Created:** 2026-07-06

## Summary

People who care about this problem rarely own one machine. This RFC adds the
**fleet layer**: a `fleet/v1` registry describing the machines an operator
owns and how to reach them, plus `frontier fleet` commands that answer "which
of my machines should run this model" and operate the existing CLI remotely
over SSH.

It also ratifies the schema treatment of a **remote node**: a second machine
reachable over a measured network link is a *generalized RFC 0002 island* —
compute + memory behind an external-class link — not a new schema concept.

## Motivation

1. **Placement across machines is the same question as placement within
   one.** The planner already ranks capacity pools by measured bandwidth
   (RFC 0002). An operator with an M5 Max, a GB10, and two Linux laptops is
   asking the identical question one level up: which resource graph fits this
   model best? The planner is pure and profile-driven, so answering it is a
   fold over registered profiles — nearly free.
2. **Remote operation is where the friction is.** Bringing up a machine today
   means SSH in, clone, install, run detect/bench, copy artifacts back. That
   is scriptable, and scripting it makes fleet bring-up (and community
   reproduction) dramatically cheaper.
3. **Two machines can be one topology.** A Thunderbolt or LAN link between
   two boxes creates a combined resource graph whose total RAM fits models
   neither machine fits alone (e.g. two 128 GB laptops vs a 239 GB artifact,
   pipeline-split at a layer boundary). Whether that is *usable* is an
   empirical question (see the dual-node spike protocol in
   [docs/spike_dual_node_thunderbolt.md](../docs/spike_dual_node_thunderbolt.md));
   this RFC only ratifies how it is *described*.

## The `fleet/v1` document

One YAML file, operator-owned, gitignored by default (`fleet/local/`) since it
contains reachability details; a sanitized example ships in `fleet/`:

```yaml
schema_version: fleet/v1
fleet_id: home_lab
machines:
- name: m5-max
  hwprofile: apple_m5_max_137gb_detected     # committed profile id
  reach: null                                 # null = this machine / manual
  roles: [bench, serve]
- name: dgx-spark
  hwprofile: gb10_128gb
  reach:
    ssh: bletort@dgx-spark                    # any ssh destination; Tailscale
    workdir: ~/Frontier-Bridge                # MagicDNS names work unchanged
  roles: [bench, serve]
- name: lenovo-4090
  hwprofile: null                             # not yet detected — fleet detect fills this
  reach: { ssh: bletort@lenovo-4090, workdir: ~/Frontier-Bridge }
  roles: [bench]
```

- `hwprofile` references a committed profile id, or `null` until `fleet
  detect` produces one.
- `reach.ssh` is an ordinary SSH destination — the fleet layer does not know
  or care that the wire is Tailscale, LAN, or localhost.
- `roles` are informative labels (`bench`, `serve`, `build`), not scheduling
  semantics.

## `frontier fleet` commands

| Verb | Behavior |
|---|---|
| `fleet plan <model>` | Run the existing planner against every registered machine's committed profile; print a ranked verdict table. Ranking is by verdict class only (`recommended` > `experimental` > `not_recommended`) — the fleet layer adds **no new scoring math**, and machines whose profile is `null` are listed as `unprofiled`, not skipped silently. |
| `fleet detect <machine>` | SSH to the machine, run `frontier detect` in its workdir, pull the emitted profile back to `hardware_profiles/local/` for review before commit. |
| `fleet bench <machine> -- <bench args>` | Same wrapper for the bench harness; results land in `results/local/`. |

Remote execution is deliberately thin: it runs the same CLI the operator
would run by hand and copies artifacts back. No daemon, no agent, no state on
the remote beyond the repo checkout.

## Remote nodes in the resource graph

RFC 0002 defined the island: compute-attached memory behind an external-class
link, holding a fixed resident expert subset. A second machine generalizes
this — and needs **no new node kinds**:

- The remote machine's compute and memory appear in a *combined* profile as
  nodes with memory `class: remote`, joined by a link with `via: ethernet`
  (or `thunderbolt` for host-to-host cable networking) carrying **measured**
  bandwidth, as always.
- Placement across the link follows the island rule: resident subsets
  computed where they live (pipeline/layer split), never per-token streaming
  of weights across the wire.
- A combined profile is a *distinct* `hwprofile/v1` document (e.g.
  `lenovo_pair_tb4`) with `provenance.method: manual` or `detect`, and plans
  against it carry the risk
  `dual_node_requires_runtime_rpc_support` until a runtime adapter
  demonstrates it and benchmark rows land.

What stays out of scope: tensor parallelism across external links (physically
unworkable at these bandwidths — the refusal math belongs in plans, not in
hopes), automatic multi-machine scheduling, and anything resembling a
cluster manager.

## What this does not do

- **No distributed-inference claims.** `fleet plan` picks *one machine per
  model*. Dual-node topologies enter only as explicit combined profiles, and
  their usability is decided by the spike protocol and verified rows —
  describable ≠ recommended (same words as RFC 0002).
- **No secrets in the repo.** Fleet files with reachability live in
  `fleet/local/` (gitignored). The committed example uses placeholder names.
- **No new numbers.** The fleet layer produces no measurements; it moves the
  existing tools closer to the machines.

## Alternatives considered

- **A daemon/agent on each machine:** richer (live telemetry, push benches),
  but a standing service is operational surface this project does not need to
  own. SSH plus the existing CLI covers the actual workflows.
- **Fleet as part of hwprofile (one graph spanning machines):** tempting —
  the schema could express it — but conflates "what exists" with "what I
  operate." Combined profiles are opt-in documents describing a deliberate
  topology (a cable someone plugged in), not an automatic union.
- **Scheduling semantics in roles:** roles as constraints ("only bench on X")
  invite a scheduler. Labels keep the human in charge of a decision that is,
  at fleet sizes of 2–5 machines, a human decision.
