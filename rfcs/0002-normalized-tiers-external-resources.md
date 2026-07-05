# RFC 0002 — Normalized Tiers and External Resources

- **Status:** Ratified
- **Author:** Brian Letort
- **Created:** 2026-07-05

## Summary

The memory hierarchy people actually own is not fixed: Thunderbolt/USB4 external SSDs, eGPU enclosures, NAS boxes, and LAN-attached machines all extend it. This RFC ratifies two things:

1. **The tier-normalization rule.** A tier is a *bandwidth class*, not a device category. The planner orders capacity pools by their **effective bandwidth to the compute node doing decode** — the minimum measured bandwidth along the link path — and assigns tiers from that order. Devices do not get tiers by what they are; they earn tiers by what they measure.
2. **Additive schema extensions** (allowed under the v1 freeze) so these devices are describable: storage classes `external_ssd` and `nas`, link vias `thunderbolt` and `usb4`, and plan tier slots `l3` and `island`.

No schema version bump. Every existing profile and plan remains valid.

## Motivation

Three examples show why category thinking fails and bandwidth normalization works:

- **Thunderbolt external SSD.** A TB5 drive moves ~3–6 GB/s through the tunnel. The M5 Max *internal* SSD measured 14.22 GB/s ([hardware_profiles/apple_m5_max_137gb_detected.yaml](../hardware_profiles/apple_m5_max_137gb_detected.yaml)). "External SSD is a storage tier below RAM" is true; *where* below depends entirely on the measured link — here, below internal storage too.
- **eGPU.** An enclosure adds a GPU with fast local VRAM behind a ~4 GB/s PCIe tunnel. Treated as a streaming tier it can *lose to the internal SSD*. Treated as a **resident island** — a fixed expert subset placed there permanently, computed where it lives, never shuttled per token — it adds real capacity. The link bandwidth decides which, and only measurement knows the link bandwidth.
- **NAS / LAN storage.** 10GbE moves ~1.2 GB/s; 100GbE moves ~12 GB/s and can legitimately outrank local SATA. Same rule, no special cases.

## The tier-normalization rule

For a hardware graph `G` and the primary compute node `c` (the GPU doing decode):

1. **Effective bandwidth** of a capacity pool `p` is the minimum measured bandwidth over the link path from `p` toward `c`'s primary memory. Unmeasured links contribute `null`.
2. **Primary pool (l0)** is the `device_local` or `unified` memory node attached to `c` over an internal-class link (`unified`, `internal`, `pcie`, `nvlink`). Memory behind an external-class link (`thunderbolt`, `usb4`, `ethernet`) is never primary while an internal candidate exists.
3. **Cache tiers (l1..lN)** are the remaining pools sorted by effective bandwidth, descending. When a pool's effective bandwidth is unmeasured it sorts by a documented class prior (system > cxl > nvme > internal_ssd > external_ssd > sata > nas > remote > other) and the plan carries the risk `tier_order_uses_class_priors_where_links_unmeasured`.
4. **Islands.** Compute-attached memory behind an external-class link (eGPU VRAM over Thunderbolt) is not a cache tier: it becomes an `island` placement — a fixed expert subset resides there permanently (`mode: resident`, `policy: static_hotlist`), sized by the island's capacity, with no per-token streaming across the link. Plans with an island carry the risk `island_placement_requires_runtime_multi_gpu_support` until a runtime adapter demonstrates it.
5. **The slowest storage pool remains the stream-on-miss backstop**, as today.

The rule is deliberately a sort, not a model: it is auditable, it degrades honestly when measurements are missing, and it produces exactly the current l0/l1/l2 behavior on the existing reference machines.

## Schema changes (additive)

`hwprofile/v1`:

- storage `class` enum adds `external_ssd`, `nas`
- link `via` enum adds `thunderbolt`, `usb4`

`plan/v1`:

- `placement.tiered.routed_experts` adds optional `l3` and `island` tier slots

`modelprofile/v1`, `benchresult/v1`: unchanged.

## What this does not do

- **No distributed inference.** A LAN-attached machine serving experts (RPC-style) is describable today (`network` nodes, `remote` memory class) but plans will not target it until a runtime adapter exists and benchmark rows prove it. Describable ≠ recommended.
- **No new detect paths yet.** `frontier detect` enumeration of Thunderbolt/USB4 devices (macOS `system_profiler`, Linux `boltctl`) is follow-on work; until then external devices enter via manual profiles with `null` for everything unmeasured.
- **No bandwidth guessing.** Class priors order tiers when links are unmeasured, but they never produce numbers: streaming math still requires measured bandwidth and stays `null` otherwise.

## Alternatives considered

- **Fixed tier slots per device category** (external SSD is always l3, NAS is always l4): simple, but wrong on real hardware — a 100GbE NAS outranks SATA, and internal NVMe outranks most Thunderbolt drives. Category assignments would encode exactly the spec-sheet guessing this project exists to replace.
- **Full path-finding over the graph** (max-flow / shortest-path effective bandwidth): more general, unnecessary at current topology sizes. The min-over-direct-links rule covers every machine shape in scope; revisit if multi-hop topologies (CXL switches, multi-node) land.
