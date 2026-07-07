# Fleet Runbook — Detecting the Reference Machines

One page per machine. Each run replaces the manual placeholder profile with a
`provenance.method: detect` profile. Commit the output after reviewing it for
anything you consider sensitive.

Machines reachable over SSH can be driven from one seat instead: register
them in a `fleet/v1` file (see [fleet/example.yaml](../fleet/example.yaml),
RFC 0005) and run `frontier fleet detect <machine>` — it executes the same
steps below remotely and pulls the profile back into
`hardware_profiles/local/` for review.

## 1. M5 Max 128 GB (macOS) — done

Already committed as
[hardware_profiles/apple_m5_max_137gb_detected.yaml](../hardware_profiles/apple_m5_max_137gb_detected.yaml).
Re-run after OS upgrades:

```bash
frontier detect -o hardware_profiles/apple_m5_max_137gb_detected.yaml
frontier validate hardware_profiles/
```

Optional: install `fio` (`brew install fio`) first — detect prefers it over the
built-in Python read bench and records the tool version.

## 2. GB10 ~120 GB (DGX Spark, Linux ARM)

```bash
git clone https://github.com/Brianletort/Frontier-Bridge.git && cd Frontier-Bridge
python3 -m venv .venv && .venv/bin/pip install -e .
sudo apt install -y fio          # preferred disk bench
.venv/bin/frontier detect -o hardware_profiles/gb10_detected.yaml
```

What to expect:

- The GPU name should match a Grace/GB10 marker, so the profile gets a single
  `unified` memory node (no separate vram/sysram). If the topology looks wrong
  (two memory nodes), the GPU name didn't match — file it, we extend
  `_UNIFIED_GPU_MARKERS` in
  [src/frontier_bridge/detect/linux_nvidia.py](../src/frontier_bridge/detect/linux_nvidia.py).
- PCIe probe: install `nvbandwidth` if available for the machine, else
  `pip install cuda-python` into the venv. Missing both is fine — the link
  ships with nulls.
- GB10 quirks are expected. Capture odd output as issues; GB10 is never the
  critical path.

## 3. RTX 6000 96 GB / 64 GB RAM / 2 TB SSD (Windows → WSL2)

One-time WSL2 setup (PowerShell as admin):

```powershell
wsl --install -d Ubuntu-24.04
```

Then create `C:\Users\<you>\.wslconfig` so WSL sees most of the host RAM —
otherwise the profile under-reports memory and the planner under-budgets:

```ini
[wsl2]
memory=56GB          # leave ~8GB for Windows
```

Restart WSL (`wsl --shutdown`), then inside Ubuntu:

```bash
# CUDA in WSL2 uses the Windows NVIDIA driver; only verify nvidia-smi works:
nvidia-smi

sudo apt update && sudo apt install -y python3-venv fio
git clone https://github.com/Brianletort/Frontier-Bridge.git && cd Frontier-Bridge
python3 -m venv .venv && .venv/bin/pip install -e .

# IMPORTANT: run from the ext4 home directory, never /mnt/c —
# the disk bench measures the filesystem it runs on, and /mnt/c (9p) numbers
# would be misleadingly slow.
.venv/bin/frontier detect -o hardware_profiles/rtx6000_wsl2_detected.yaml
```

What to expect:

- `provenance.virtualization: wsl2` is recorded automatically; the profile id
  ends in `_wsl2_detected`.
- The RAM figure is what `.wslconfig` assigned, not the host total — that is
  the truth for anything running inside WSL2, which is where the runtimes live.
- The GPUDirect Storage link is recorded `available: false` (GDS does not work
  under WSL2); expert streaming there uses the pinned-RAM bounce path.
- Model storage: keep GGUFs on the ext4 filesystem (e.g. `~/models`), not
  `/mnt/c`. Budget: GLM-5.2 Q2 (~250 GB class) + DeepSeek V4 Flash fit in 2 TB
  with working room.

## 4. Lenovo pair (Ubuntu, i9 / 128 GB RAM / 16 GB VRAM: RTX 5000 Ada and RTX 4090)

Both boxes run Ubuntu natively — no WSL2 caveats. Same steps as the GB10
(clone, venv, `fio`, detect), expecting the **discrete** topology: separate
`vram` memory node, PCIe link, GDS link recorded with its availability.

- These are the first real runs of the discrete-NVIDIA detect path
  (fixture-tested until now). File whatever breaks.
- Detect both machines even though they differ only by GPU — the RTX 5000 Ada
  vs RTX 4090 pair is a controlled comparison the matrix can use.
- With the 5090 eGPU attached: detect currently classifies all GPUs as
  internal PCIe. The Thunderbolt-attached GPU needs per-GPU link
  classification (sysfs TB topology or a measured per-GPU bandwidth probe)
  emitting the `thunderbolt` link `via` from RFC 0002 — until that lands,
  hand-edit the detected profile's link and mark the field's provenance
  accordingly.
- The dual-node bridge experiment for this pair has its own protocol:
  [spike_dual_node_thunderbolt.md](spike_dual_node_thunderbolt.md).

## After each run

```bash
frontier validate hardware_profiles/
git add hardware_profiles/ && git commit -s -m "Add detected profile for <machine>"
```

Review the YAML before committing: hostnames or serials do not belong in
profiles. Once a detected profile lands, delete the manual placeholder it
replaces.
