# Launch Checklist

The repo goes public when this checklist is complete — not before. Two gates, per the roadmap: **G0** (cleared to exist publicly) and **G1** (cleared to promote).

## Gate G0 — Legal, brand, and IP sign-off

Nothing public before G0.

- [ ] Legal review of the Apache 2.0 release (plus sponsorship attribution review, only if a sponsor is confirmed)
- [ ] Sponsor comms/brand sign-off on any sponsorship language (only applicable once a sponsor is confirmed in writing — no attribution before then)
- [ ] IP review checkpoint #1 complete: confirm the open/held split in [IP_NOTICE.md](../IP_NOTICE.md) (schemas, profiles, recipes, harness, basic LRU/hotlist/LFU policies → open; router-aware prefetch, workload hotlists, adaptive session cache, agent-loop persistence → held)
- [ ] Name clearance: trademark + GitHub/PyPI/domain search for "Frontier Bridge"; backup name selected in case clearance fails
- [ ] Sponsorship confirmation in writing (no sponsor is named anywhere in the repo until this exists)
- [ ] License headers / NOTICE file final

## Gate G1 — Launch with real numbers

The repo can be public but quiet after G0. Promotion waits for G1.

- [ ] `frontier detect` verified on all three reference machines (RTX 6000, GB10, M5 Max) — currently only the M5 Max path has run on real hardware
- [ ] Model profiles pinned: exact GGUF URLs + sha256 hashes for GLM-5.2 Q2/Q4 routed and DeepSeek V4 Flash Q2/Q4; DeepSeek parameter counts verified upstream (planner currently refuses it by design)
- [ ] Benchmark harness (`frontier bench`) shipped, producing `benchresult/v1`
- [ ] Compatibility matrix has real rows: every published number reproduced twice, per the results-integrity rule
- [ ] Demo ready: `detect → plan → run → bench` live on the RTX 6000, ending with a coding agent hitting the local OpenAI-compatible endpoint
- [ ] Community scaffolding: benchmark submission PR template, "good first profile" issues, GitHub Discussions enabled

## Launch sequence (after G1)

1. Repo public on GitHub
2. LocalLLaMA tester post — non-corporate voice, asking for RTX 6000 / 5090 / GB10 / M-series / Strix Halo testers and benchmark JSON submissions
3. LinkedIn post — open engineering contribution to the AI infrastructure continuum
4. Hacker News — only after a demo video and reproducible numbers exist; HN punishes vapor

## Publishing mechanics

- Push `main` to `github.com/Brianletort/Frontier-Bridge` (history is clean — no secrets, no large artifacts; model weights are referenced by hash, never committed)
- Tag `v0.1.0`
- Enable Discussions, issue labels by hardware class, branch protection on `main`
