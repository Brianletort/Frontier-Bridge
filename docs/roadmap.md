# Roadmap — Execution Plan to G1 and Beyond

**Objective:** prove the hypothesis — near-frontier MoE models (GLM-5.2 at
744B total / 40B active, DeepSeek V4 Flash at 284B / 13B) are practically
runnable on the 96 GB VRAM / 128 GB unified / fast-DDR+NVMe hardware class —
with hash-pinned, twice-reproduced numbers, then launch.

Launch gates (G0/G1) are defined in [launch_checklist.md](launch_checklist.md).
Per-machine bring-up steps live in [fleet_runbook.md](fleet_runbook.md); the
exact benchmark loop lives in [benchmark_playbook.md](benchmark_playbook.md).

## Phase 0 — M5 Max 128 GB (now)

The machine already profiled
([apple_m5_max_137gb_detected](../hardware_profiles/apple_m5_max_137gb_detected.yaml))
can produce the first verified matrix row with no other hardware.

1. **Start the model downloads first** — they are the long pole, not the
   benchmarks:
   - DeepSeek V4 Flash Q2 (107 GB) → M5 Max. First verified-row candidate:
     fits resident in unified memory, planner verdict `recommended`
     ([plans/m5max_dsv4flash_q2_chat_16k.yaml](../plans/m5max_dsv4flash_q2_chat_16k.yaml)).
   - Queue GLM-5.2 Q2 (239 GB) for the RTX 6000 box.
2. **First verified row** (M5 Max × DeepSeek V4 Flash Q2 × coding_agent × 32K):

   ```bash
   frontier plan deepseek-v4-flash --hardware apple_m5_max_137gb_detected \
     --workload coding_agent --ctx 32768 -o plans/m5max-dsv4-q2-coding-32k.yaml
   frontier run plans/m5max-dsv4-q2-coding-32k.yaml --model-path ~/models/dsv4-flash/...
   frontier bench --plan plans/m5max-dsv4-q2-coding-32k.yaml \
     --suite coding_agent --runtime-commit <llama.cpp commit>
   # two reproductions with fresh runtime starts, then:
   frontier results matrix -o docs/compatibility_matrix.md
   ```

   This exercises the entire pipeline end to end and makes the
   "four pins + two reproductions" rule real before new hardware enters
   the picture.

## Phase 1 — RTX 6000 96 GB (at home, priority 1)

The launch-demo machine and the direct test of the 96 GB VRAM gap.

1. WSL2 bring-up per [fleet_runbook.md](fleet_runbook.md): Ubuntu 24.04,
   `.wslconfig` memory assignment, `nvidia-smi` check, `fio`, detect **from
   the ext4 home directory** (never `/mnt/c`).
2. Validate the detected profile — this is the Linux/NVIDIA detect path's
   first run on real hardware; expect and file surprises. Confirm
   `virtualization: wsl2` and GDS `available: false` are recorded.
3. Build llama.cpp CUDA; record the commit.
4. Tiered run: GLM-5.2 Q2 (239 GB) across VRAM + RAM + NVMe. Plan tiers map
   to llama.cpp flags: L0 → `-ngl` + GPU-resident expert overrides,
   L1 → `--n-cpu-moe N`, L2 → mmap paging. Capture `ssd_total_read_gb` —
   the SSD-endurance data before the community asks.
5. Produce the launch-demo row: coding_agent at 32K (stretch: 128K),
   reproduced twice, ending with a coding agent hitting the local
   OpenAI-compatible endpoint. This row is the hypothesis test.
6. Also bench DeepSeek V4 Flash Q2 on this box — 107 GB does not fit 96 GB
   VRAM, so it is a genuine hybrid VRAM+RAM data point, cheap to add while
   the machine is set up.

**Decision point:** if tuned baseline decode on GLM-5.2 Q2 is unusable
(below a `usable` rating), that is a finding, not a failure — publish it
honestly; it feeds the Phase 4 spike's motivation directly.

## Phase 2 — GB10 128 GB (at home, priority 2, timeboxed)

GB10 is never the critical path. Timebox it to one session.

1. `frontier detect` per the runbook; verify the unified-memory topology
   (single `unified` node — two memory nodes means the GPU-name marker did
   not match; file it).
2. llama.cpp CUDA build for SM121; pin the commit.
3. One row: DeepSeek V4 Flash Q2 resident, coding_agent 32K. GLM-5.2 Q2
   unified+SSD if time allows.
4. Capture quirks as issues; do not chase them.

## Phase 3 — Matrix, gates, and launch

1. Fold all rows into `docs/compatibility_matrix.md`
   (`frontier results matrix`); usability labels never better than
   `usability_suggested`.
2. **G0 paperwork — start now, in parallel with everything above** — it has
   weeks of latency and zero code dependency: Digital Realty legal sign-off,
   comms/brand review, IP review checkpoint #1, name clearance with a backup
   name, sponsorship confirmation in writing. G0 is the plan's only external
   dependency and its biggest schedule risk.
3. Community scaffolding: benchmark-submission PR template,
   "good first profile" issues, GitHub Discussions.
4. Launch sequence as written in
   [launch_checklist.md](launch_checklist.md): repo public → LocalLLaMA
   tester post → Digital Realty LinkedIn → Hacker News only after a demo
   video and reproducible numbers exist.

## Phase 4 — Post-launch: CUDA expert-streaming spike (2 weeks, RTX 6000)

The core scientific question: does an explicit NVMe → pinned-RAM → VRAM
expert cache beat mmap + page cache? Protocol and decision rule are already
written in
[spike_cuda_expert_streaming.md](spike_cuda_expert_streaming.md):
days 1–2 tuned baseline (publishable regardless), days 3–4 expert-access
traces (the go/no-go ceiling check), days 5–8 smallest-slice prototype,
days 9–10 head-to-head. Adopt only at ≥1.5x decode tps or ≥2x p95 at equal
VRAM budget. Open policies only; router-aware prefetch stays held per
[IP_NOTICE.md](../IP_NOTICE.md).

## Known gaps (non-blocking for G1)

- **No RTX 5090 / 128 GB DDR profile** despite being in the README target
  table — the "fast DDR + fast SSD, modest VRAM" class is unrepresented.
  Cheapest fix: recruit a community profile at launch via a
  "good first profile" issue.
- **Strix Halo** is named in the launch post plan but has no profile or
  detect path — same treatment, community-sourced.

## Sequencing summary

| When | What | Output |
|---|---|---|
| Now | M5 Max × DeepSeek V4 Flash Q2 verified row; start downloads | First real matrix row; pipeline proven |
| Home, day 1 | RTX 6000 WSL2 bring-up + detect | 2nd reference profile; Linux detect validated |
| Home, day 1–2 | GLM-5.2 Q2 tiered run + bench ×3 | Launch-demo row; hypothesis data |
| Home, day 2 | GB10 detect + one row (timeboxed) | 3rd profile; G1 detect gate closed |
| Parallel | G0 legal/brand/IP/name/sponsorship | Clearance to promote |
| After G1 | Launch sequence | Public repo + community pipeline |
| Post-launch | Expert-streaming spike | Adopt/reject explicit expert cache |

Net: two solid days of hands-on-hardware work stand between the current
state and G1's technical gates. The schedule risk is entirely G0 — the
legal and sponsorship threads should move this week so the paperwork lands
when the numbers do.
