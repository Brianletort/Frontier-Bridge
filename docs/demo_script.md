# CTO Demo Script — detect → plan → run → bench

Target duration: 8 minutes live, or a 4-minute recording. Everything below is
real commands against real hardware; nothing staged. Record with the terminal
font large and the plan YAML on screen long enough to read the verdict.

## Setup (before the demo)

- Model downloaded and hash-verified on the demo machine.
- Runtime installed (llama.cpp) with the commit hash noted.
- `frontier` installed in a venv; repo checked out clean.

## Beat 1 — "What is this machine, really?" (60s)

```bash
frontier detect
```

Talking points while it prints:
- Nodes and links, measured — the SSD number is a live read benchmark, not a
  spec sheet. Unknown values are null; the tool never guesses.
- Same schema describes this laptop, a GB10, an RTX 6000 box, or a rack node.

## Beat 2 — "Can I run a 744B model here?" (90s)

```bash
frontier plan glm-5.2 --hardware <detected_profile> --workload coding_agent --ctx 32768
```

Talking points:
- 744B total, but only ~40B active per token — the planner tiers 446 GB of
  routed experts across memory and SSD and keeps the dense core resident.
- Point at `expected.streaming`: worst-case miss cost computed from measured
  GGUF header sizes and the measured SSD bandwidth. Physics, not marketing.
- Then the counter-demo — the refusal:

```bash
frontier plan glm-5.2 --hardware <detected_profile> --workload long_context --ctx 2000000
```

`verdict: not_recommended`, machine-readable reasons. A planner you can trust
when it says yes is one that says no out loud.

## Beat 3 — "Run it." (2 min)

```bash
frontier run plans/<demo-plan>.yaml --model-path ~/models/<model>.gguf
```

- sha256 verification against the pinned profile, then launch, then the
  health-check. When "Endpoint ready" prints, hit it live:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"One sentence: why do MoE models change local AI?"}]}'
```

- Optional agent moment: point Cursor / any OpenAI-compatible client at
  `http://127.0.0.1:8080/v1` and ask it to write a function.

## Beat 4 — "Prove it." (90s)

```bash
frontier bench --plan plans/<demo-plan>.yaml --suite coding_agent --port 8080
frontier results matrix
```

Talking points:
- One JSON per run: TTFT, decode tps, p95/p99 latency, memory peaks, total
  SSD bytes read. Community submissions use the same file format — the
  leaderboard is a fold, not a database.
- `status: claimed` until all four pins and two reproductions exist. The tool
  enforces the trust model.

## Close (30s)

The continuum line: same profiles, plans, and benchmark standard from this
workstation up through lab servers, private AI racks, and colocation or
cloud infrastructure — moving up the bridge is a schema change, not a
rewrite.
