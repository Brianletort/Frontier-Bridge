# Linux / NVIDIA detect verification

Status of `frontier detect` on Linux, and the checklist for verifying it on
hardware we have not run yet. The rule is the same as everywhere else in this
repo: nothing is called verified until it has run for real.

## What has been verified

| Path | Status | Evidence |
|---|---|---|
| CPU / RAM / os-release parsing (x86_64 Linux) | verified | Runs in Linux containers; emits schema-valid `hwprofile/v1` (`nogpu_*_detected`) |
| lscpu-absent fallback (`os.cpu_count`) | verified | Minimal container images without util-linux |
| nvidia-smi / lsblk / meminfo parsers | fixture-tested | `tests/test_detect_parsers.py` |
| NVIDIA discrete GPU (vram node, PCIe links) | **not yet run on real hardware** | fixtures only |
| WSL2 detection and GDS-unavailable marking | **not yet run on real hardware** | fixtures only |
| GB10/Grace unified-memory topology | **not yet run on real hardware** | fixtures only |

## Checklist for the first real NVIDIA run

Run on the target box (bare metal or WSL2 from a native ext4 path — never
`/mnt/c`):

```bash
pip install -e ".[dev]"
frontier detect -o hardware_profiles/local/$(hostname).yaml
frontier validate hardware_profiles/local/$(hostname).yaml
```

Then check, in the emitted YAML:

1. One `gpu<i>` compute node per physical GPU, with the right `model` and `sm`.
2. One `vram<i>` node per GPU whose `capacity_gb` matches `nvidia-smi -q -d MEMORY`
   (decimal GB, so a 96GiB card reads ~102.6).
3. `sysram0.capacity_gb` matches `MemTotal` (WSL2: matches the `.wslconfig`
   assignment, not host RAM — expected, and `provenance.virtualization: wsl2`).
4. One `nvme<j>` node per NVMe disk; `measured.seq_read_gbps` is plausible for
   the drive (fio preferred; install it for the qd/iops fields).
5. PCIe link `measured.h2d_gbps`/`d2h_gbps` filled when `nvbandwidth` or
   `cuda-python` is present, null otherwise (null is correct, not a bug).
6. On WSL2, the `gds` link says `available: false`; on bare metal it says
   `"unknown"` until GDS is actually probed.
7. On GB10/Grace-class machines: a single `unified0` memory node and no
   `vram` nodes.

If all seven hold, open a PR adding the profile under `hardware_profiles/` and
flip the corresponding row in the table above with the PR as evidence. If any
fail, file an issue with the raw collector outputs (`nvidia-smi --query-gpu=...`,
`lsblk -dn -o NAME,TYPE,SIZE,TRAN -b`, `/proc/meminfo`).
