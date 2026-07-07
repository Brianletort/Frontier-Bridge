# Roadmap — Execution Plan to G1 and Beyond

**Objective:** prove the hypothesis — near-frontier MoE models (GLM-5.2 at
744B total / 40B active, DeepSeek V4 Flash at 284B / 13B) are practically
runnable on the 96 GB VRAM / 128 GB unified / fast-DDR+NVMe hardware class —
with hash-pinned, twice-reproduced numbers, then launch. The deliverable is
the **runbook** ([RFC 0003](../rfcs/0003-runbooks.md)): one distributable
playbook per hardware class, numbers folded from verified rows.

Launch gates (G0/G1) are defined in [launch_checklist.md](launch_checklist.md).
Per-machine bring-up steps live in [fleet_runbook.md](fleet_runbook.md); the
exact benchmark loop lives in [benchmark_playbook.md](benchmark_playbook.md).

**Resequenced 2026-07-06** to match the fleet actually on hand: GB10 is
reachable now (Tailscale) and promoted to the next machine; two Ubuntu
laptops (i9 / 128 GB RAM / 16 GB VRAM — RTX 5000 Ada and RTX 4090) plus a
5090 eGPU enclosure join as the discrete-NVIDIA proving ground and the
dual-node spike pair; the RTX 6000 (Windows → WSL2) phase is deferred, not
changed — its runbook becomes the fourth. Paper tracks (RFCs 0003–0005,
runbook CLI, fleet CLI, catalog audit) landed ahead of the hardware work.

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

## Phase 1 — GB10 128 GB over Tailscale (now, priority 1)

Promoted from Phase 2: the machine is reachable today and its class shares
the `unified-128gb` runbook with the M5 Max.

1. `frontier fleet detect dgx-spark` (or the manual steps in
   [fleet_runbook.md](fleet_runbook.md) §2); verify the single `unified`
   memory node; commit the detected profile, delete the manual placeholder.
2. DeepSeek V4 Flash Q2 resident: bench ×3 (chat + coding_agent), fold the
   matrix, add the class's second `known_profile` to the runbook.
3. GLM-5.2 Q2 plan (239 GB vs 128 GB unified): commit whatever verdict the
   planner returns — a refusal is runbook material, not a failure.

## Phase 1b — Lenovo pair (Ubuntu, discrete NVIDIA)

First real run of the `linux_nvidia.py` discrete-GPU detect path, and the
`gpu-laptop-128gb` runbook's evidence base.

1. Detect both boxes (RTX 5000 Ada vs RTX 4090, otherwise identical — a
   natural controlled comparison).
2. Tiered rows on the 4090 box: DSV4 Q2 (`--n-cpu-moe`, experts in RAM) and
   GLM-5.2 Q2 (NVMe tier engaged).
3. 5090 eGPU attached: extend detect with Thunderbolt GPU link
   classification, commit the island-topology profile, bench the resident
   island (the empirical test of RFC 0002's island claim).
4. Dual-node Thunderbolt bridge spike per
   [spike_dual_node_thunderbolt.md](spike_dual_node_thunderbolt.md) —
   adopt as a supported topology or publish the refusal, by the written
   decision rule.

## Phase 2 (deferred, unchanged) — RTX 6000 96 GB via WSL2

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

## Phase 3 — Matrix, runbooks, gates, and launch

1. Fold all rows into `docs/compatibility_matrix.md`
   (`frontier results matrix`); usability labels never better than
   `usability_suggested`.
2. **G0 paperwork — start now, in parallel with everything above** — it has
   weeks of latency and zero code dependency: legal review, IP review
   checkpoint #1, name clearance with a backup name, and — only if a sponsor
   comes aboard — sponsorship confirmation in writing before any attribution.
   G0 is the plan's only external dependency and its biggest schedule risk.
3. Runbooks per benchmarked class (`frontier runbook verify` green in CI):
   `unified-128gb` updated with GB10 rows, `gpu-laptop-128gb` authored from
   the Lenovo rows, `rtx6000-96gb` after Phase 2.
4. Community scaffolding: benchmark-submission PR template,
   "good first profile" issues, runbook contribution guide, GitHub
   Discussions.
5. Launch sequence as written in
   [launch_checklist.md](launch_checklist.md): repo public → LocalLLaMA
   tester post → LinkedIn → Hacker News only after a demo
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
| Done | M5 Max × DeepSeek V4 Flash Q2 verified rows | First real matrix rows; pipeline proven |
| Done (paper) | RFCs 0003–0005; runbook + fleet CLI; catalog audit with sourced licenses; first runbook | The product standard, ahead of the hardware data |
| Now | GB10 detect over Tailscale + DSV4 rows + GLM verdict | 2nd benchmarked machine; unified-128gb runbook strengthened |
| Next | Lenovo pair detect; 4090-box tiered rows; eGPU island | Discrete detect validated; gpu-laptop runbook material |
| Next | Dual-node Thunderbolt spike (5 days, written decision rule) | Supported topology or published refusal |
| Deferred | RTX 6000 WSL2 bring-up + GLM-5.2 launch-demo row | The 96 GB VRAM hypothesis test; 4th runbook |
| Parallel | G0 legal/brand/IP/name/sponsorship | Clearance to promote — now covers the runbook brand too |
| After G1 | Launch sequence | Public repo + community pipeline |
| Post-launch | Expert-streaming spike | Adopt/reject explicit expert cache |

The schedule risk is entirely G0 — the legal and sponsorship threads should
move this week so the paperwork lands when the numbers do.
