# The Enterprise Bridge

Frontier Bridge exists because AI infrastructure is becoming heterogeneous. Models will run in hyperscale clouds, enterprise data centers, colocation facilities, edge deployments, and increasingly on powerful workstations. No single tier wins; the value is in moving fluidly between them.

## The continuum

```text
High-end consumer / workstation
  ↓
Single-user local AI
  ↓
Team workstation / lab server
  ↓
Department GPU server
  ↓
Enterprise private AI rack
  ↓
Colocation / connected AI infrastructure
  ↓
Cloud, NSP, CSP, partner ecosystem
```

A typical journey up the bridge:

| Stage | Hardware | What Frontier Bridge provides |
|---|---|---|
| Experiment | M5 Max 128 GB | detect, plan, run a frontier MoE locally |
| Commit | RTX 6000 96 GB workstation | tiered VRAM/RAM/NVMe plans, reproducible benchmarks |
| Scale to team | Multi-GPU lab server | same schemas — more nodes and links in the resource graph |
| Productionize | Private AI rack in colocation | enterprise-bridge profiles, deployment recipes |
| Connect | Multi-site, cloud-adjacent, partner ecosystem | network nodes/links, multi-node plans |

The design point that makes this work: **the hardware profile schema is a resource graph, not a fixed field set** (see [RFC 0001](../rfcs/0001-resource-graph-schemas.md)). A rack node is the same schema as a laptop — more compute nodes, more memory classes, more links. Moving up the bridge is a schema change, not a rewrite. Your plans, benchmarks, and workload profiles carry forward.

## Why Digital Realty

Digital Realty sponsors Frontier Bridge because it already operates the upper tiers of this continuum: a global platform of 300+ data centers across 55+ metros serving 5,000+ customers, with interconnection, cloud on-ramps, service-provider ecosystems, and support for deployments from single racks to private suites and high-density AI workloads ([digitalrealty.com](https://www.digitalrealty.com/)).

The sponsorship story, plainly:

> Digital Realty is not trying to replace model providers or inference runtimes. It is helping build the missing infrastructure layer — hardware profiles, memory planning, runtime selection, benchmarks, and deployment recipes — that helps developers and enterprises get the most from the infrastructure they already have, and a credible path to enterprise-grade infrastructure when they outgrow it.

## What lands here (Phase 6+)

Enterprise-bridge profiles, in the same `hwprofile/v1` schema:

- multi-GPU workstation
- rack-mounted GPU server
- colocation private AI node
- multi-site inference (network links become load-bearing)
- cloud-adjacent burst
- CSP/NSP-connected inference

These start as schema + documentation exercises with deployment recipes, and graduate to verified rows in the compatibility matrix as benchmark results land — the same claimed-vs-verified rule as everything else in this repo.

## What we will not claim

The bridge is a path, not a pitch. Nothing in this repo will assert that a given deployment tier is right for you — the planner's job is to tell you honestly what your current infrastructure can do, and what the next tier would buy you, with measured numbers on both sides.
