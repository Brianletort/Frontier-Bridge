# Benchmark Playbook — Running the v0.1 Matrix

The exact, repeatable procedure for producing matrix rows on the three
reference machines. Every number gets reproduced twice before publication —
the tool enforces it (`status: verified` requires all four pins plus two
reproductions).

## The loop (any machine)

```bash
# 1. Plan (uses the machine's detected profile)
frontier plan glm-5.2 --hardware <profile_id> --workload coding_agent --ctx 32768 \
  -o plans/<machine>-glm52-q2-coding-32k.yaml

# 2. Run (verifies sha256 against the profile pins, launches, health-checks)
frontier run plans/<machine>-glm52-q2-coding-32k.yaml \
  --model-path ~/models/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf

# 3. Bench (separate terminal; runtime commit hash pins the build)
frontier bench --plan plans/<machine>-glm52-q2-coding-32k.yaml \
  --suite coding_agent --port 8080 \
  --runtime-commit "$(git -C ~/src/llama.cpp rev-parse HEAD)" \
  --runtime-pid <pid>

# 4. Reproduce (same plan, fresh runtime start, twice)
frontier bench ... --repro-of <result_id_1> --repro-of <result_id_2>

# 5. Fold into the matrix
frontier results matrix -o docs/compatibility_matrix.md
```

Move results worth publishing from `results/local/` (gitignored) into
`results/community/` or `results/verified/` via PR.

## Model staging

| Artifact | Size | Where it fits |
|---|---|---|
| GLM-5.2 UD-IQ2_M (q2_routed) | 239 GB | M5 Max (8TB), RTX box ext4 (2TB), GB10 |
| GLM-5.2 UD-Q4_K_XL (q4_routed) | 467 GB | M5 Max only (v0.1) |
| DeepSeek V4 Flash Q2_K-XL (q2_imatrix) | 107 GB | all three |
| DeepSeek V4 Flash Q4_K_M-XL (q4_imatrix) | 175 GB | all three |

Download with the Hugging Face CLI (repos and quant dirs are pinned in
`model_profiles/`):

```bash
hf download unsloth/GLM-5.2-GGUF --include "UD-IQ2_M/*" --local-dir ~/models/glm-5.2
hf download teamblobfish/DeepSeek-V4-Flash-GGUF --include "Q2_K-XL/*" --local-dir ~/models/dsv4-flash
```

`frontier run` verifies the shard against the profile's sha256 pin before
launching. Do not skip verification for publishable runs.

## Per-machine notes

### M5 Max 128 GB (macOS)

- Runtime: llama.cpp Metal (`brew install llama.cpp` or build from source —
  record the commit either way).
- DeepSeek V4 Flash Q2 (107 GB) fits resident: expect the planner's
  `recommended` verdict. This is the first verified-row candidate.
- GLM-5.2 Q2 (239 GB) exceeds unified memory: use llama.cpp MoE offload
  (`--n-cpu-moe` / `-ot 'exps=CPU'`-style tensor overrides, per the plan's
  tier budgets) and expect `experimental`.
- powermetrics wants sudo; without it power fields stay null. Fine.

### GB10 ~120 GB (DGX Spark)

- Runtime: llama.cpp CUDA (build for SM121) and/or ds4-zgx. Pin commits.
- Same model set as the M5 Max; unified-memory plans come out with a single
  tier. GB10 is second in every queue — capture quirks as issues, don't chase.

### RTX 6000 96 GB / 64 GB RAM (WSL2)

- Everything inside WSL2 ext4 (`~/models`, never `/mnt/c`).
- Runtime: llama.cpp CUDA. The plan's three tiers map to llama.cpp flags:
  L0 (VRAM budget) → `-ngl` + expert tensor overrides kept on GPU;
  L1 (RAM) → `--n-cpu-moe N` (N from the plan's l1 expert_layer_capacity);
  L2 (NVMe) → mmap paging (default) — record `ssd_total_read_gb` from bench.
- This machine produces the launch-demo row: GLM-5.2 Q2 coding_agent 32K–128K.

## Publication bar (from GOVERNANCE.md)

1. All four pins non-null: plan hash, model sha256, runtime commit, hwprofile.
2. Two reproductions listed in the result.
3. The usability label in the README matrix is a maintainer decision backed by
   `usability_suggested` — never better than the suggestion, only equal or
   more conservative.
