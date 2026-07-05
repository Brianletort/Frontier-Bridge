# IP Notice

This repository is licensed under Apache 2.0, which includes an explicit patent grant for everything published here.

## What is open now

The following components are published and fully open:

- The four v1 schemas (`hwprofile`, `modelprofile`, `plan`, `benchresult`) and the RFC process around them.
- Hardware and model profiles, recipes, and compatibility data.
- The `frontier detect` profiler and `frontier plan` rules-based planner v0.
- Basic, well-known cache policies referenced by plans: **layer-aware LRU**, **static hotlist preload**, and **LFU**.
- The benchmark harness and result schema (when shipped).

## What is deliberately not published yet

Certain scheduling and caching techniques are under internal IP review before any public disclosure (code or design docs):

- Router-aware expert prefetch (using model routing behavior to anticipate SSD/RAM loads).
- Workload-aware expert hotlists (per-workload expert residency profiles).
- Adaptive session caching and agent-loop cache persistence (reusing expert/KV state across multi-step tasks).
- Tail-latency (p95/p99) schedule optimization.

Plans emitted by this repo reference only the open policies above. If a plan field accepts a policy name, only open policy names are valid values in v1.

This split is a process safeguard, not a statement that any of the above is patentable or will be patented. When review completes, cleared techniques will be published here under the same Apache 2.0 terms.

## Trademarks

"Frontier Bridge" name clearance is in progress. No trademark rights are granted by the Apache 2.0 license (see Section 6).
