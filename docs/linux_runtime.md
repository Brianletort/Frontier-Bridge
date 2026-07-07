# Linux Runtime Setup — CUDA llama.cpp with a Recorded Commit

The Linux + NVIDIA path builds llama.cpp from source so the CUDA backend
matches your driver and the build is pinned to an exact commit — the pin
`benchresult/v1` requires before any row can be `verified`.

## Build

```bash
# prerequisites (Ubuntu)
sudo apt install -y build-essential cmake git
# CUDA toolkit (nvcc) from https://developer.nvidia.com/cuda-downloads —
# nvidia-smi alone is the driver, not the toolkit.

./scripts/build_llama_cpp_cuda.sh            # builds master, resolves + records the sha
./scripts/build_llama_cpp_cuda.sh b1234      # or any tag/branch/sha you choose
```

The script installs `llama-server` into `~/.local/bin` and writes the exact
commit it built to `~/.local/bin/llama_cpp_commit.txt`. That sha is the
runtime pin:

```bash
frontier bench --plan <plan.yaml> --suite coding_agent \
    --runtime-commit "$(cat ~/.local/bin/llama_cpp_commit.txt)"
```

Reproductions must be built from the same recorded commit — record it, don't
remember it.

## How plan tiers map to llama.cpp flags

The planner's tier slots translate to launch flags (the plan's `runtime.launch`
carries the exact command; this is what it means):

| Plan tier | llama.cpp mechanism |
|---|---|
| `l0` (VRAM) | `-ngl` layer offload + GPU-resident expert overrides |
| `l1` (system RAM) | `--n-cpu-moe N` — routed experts computed on CPU from RAM |
| `l2` (NVMe, `stream_on_miss`) | mmap paging — the OS page cache is the cache |

Sweeping `--n-cpu-moe` to find the stability boundary on a new machine is a
built-in experiment:

```bash
frontier bench sweep-offload --model ~/models/<gguf> --values 8,16,24,32 \
    --output results/local/offload_sweep.jsonl
```

## Before benching a new machine

1. `frontier doctor` — readiness with fixes.
2. `frontier detect -o hardware_profiles/local/<name>.yaml` — run it from the
   filesystem the models will live on; the disk bench measures what it runs on.
3. `frontier bench ssd-stream` — records the expert-slice random-read floor
   (`expert_read_gbps`) the streaming math uses; sequential numbers flatter
   the drive.

The full loop from plan to verified row is the
[benchmark playbook](benchmark_playbook.md).
