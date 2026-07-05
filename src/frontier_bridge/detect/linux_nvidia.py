"""Linux / NVIDIA detection, including WSL2 and GB10/Grace unified memory.

Collectors: nvidia-smi, /proc/meminfo, lscpu, lsblk, fio (preferred) or a
bounded uncached Python read benchmark, and a PCIe H2D/D2H probe
(nvbandwidth, else cuda-python pinned memcpy, else nulls).

Environment handling:

- WSL2 is detected from /proc/version or WSL_DISTRO_NAME and recorded in
  ``provenance.virtualization``. Under WSL2 the RAM figure is what WSL was
  assigned (.wslconfig), not host RAM, and disk numbers reflect ext4-on-VHDX —
  run detection from a native ext4 path, never /mnt/c (9p is slow and would
  produce misleading numbers). GPUDirect Storage is unavailable under WSL2;
  the gds link is emitted with ``available: false`` there.
- GB10 / Grace-class systems (aarch64 + NVIDIA GPU with coherent memory) are
  expressed as one ``unified`` memory node, like the macOS path — same schema,
  different link topology.

Verification status: the CPU/RAM/disk path has run on real Linux (containers,
x86_64) and emits schema-valid profiles; the NVIDIA GPU path is fixture-tested
but has not yet run against a physical NVIDIA GPU. First runs on real GPU
hardware should follow docs/linux_verification.md and contribute the resulting
profile.
"""

from __future__ import annotations

import os
import platform
from typing import Any

from frontier_bridge.detect.common import (
    disk_read_bench,
    is_wsl2,
    probe_pcie_bandwidth,
    run_command,
    sanitize_id,
    utc_now_iso,
)

# GPU-name markers for Grace/GB10-class coherent unified memory. Extend as
# verified hardware lands; unknown systems default to discrete vram+sysram.
_UNIFIED_GPU_MARKERS = ("gb10", "grace", "gb200", "gh200")

_NULL_SSD_MEASURED: dict[str, Any] = {
    "seq_read_gbps": None,
    "rand_read_4k_iops": None,
    "qd_used": None,
    "bench_tool": None,
}

_NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version,compute_cap",
    "--format=csv,noheader,nounits",
]


def parse_nvidia_smi(output: str | None) -> list[dict[str, Any]]:
    """Parse `nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap
    --format=csv,noheader,nounits` output into GPU dicts (one per line)."""
    gpus: list[dict[str, Any]] = []
    if not output:
        return gpus
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name, mem_mib, driver, compute_cap = parts[0], parts[1], parts[2], parts[3]
        try:
            vram_gb: float | None = round(float(mem_mib) * 1024 * 1024 / 1e9, 1)
        except ValueError:
            vram_gb = None
        gpus.append(
            {
                "name": name or None,
                "vram_gb": vram_gb,
                "driver_version": driver or None,
                "compute_cap": compute_cap or None,
            }
        )
    return gpus


def parse_meminfo(text: str | None) -> float | None:
    """Extract MemTotal in GB from /proc/meminfo content."""
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                return round(int(fields[1]) * 1024 / 1e9, 1)
    return None


def parse_lscpu(text: str | None) -> dict[str, Any]:
    """Extract core count, model name, and NUMA node count from `lscpu` output."""
    info: dict[str, Any] = {"cores": None, "model": None, "numa_nodes": None}
    if not text:
        return info
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "CPU(s)" and value.isdigit():
            info["cores"] = int(value)
        elif key == "Model name":
            info["model"] = value
        elif key == "NUMA node(s)" and value.isdigit():
            info["numa_nodes"] = int(value)
    return info


def parse_os_release(text: str | None) -> str | None:
    """Extract PRETTY_NAME from /etc/os-release content."""
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.partition("=")[2].strip().strip('"') or None
    return None


def parse_lsblk(text: str | None) -> list[dict[str, Any]]:
    """Parse `lsblk -dn -o NAME,TYPE,SIZE,TRAN -b` output into disk dicts.

    Only nvme-transport disks are returned; others are out of scope for v0.
    """
    disks: list[dict[str, Any]] = []
    if not text:
        return disks
    for line in text.strip().splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        name, dev_type, size_bytes, tran = fields[0], fields[1], fields[2], fields[3]
        if dev_type != "disk" or tran != "nvme":
            continue
        try:
            capacity_gb: float | None = round(int(size_bytes) / 1e9, 1)
        except ValueError:
            capacity_gb = None
        disks.append({"name": name, "capacity_gb": capacity_gb})
    return disks


def is_unified_memory_system(gpus: list[dict[str, Any]], machine: str) -> bool:
    """Heuristic for Grace/GB10-class coherent unified memory: aarch64 plus a
    GPU whose name carries a known marker. Discrete is the safe default."""
    if machine != "aarch64":
        return False
    for gpu in gpus:
        name = (gpu.get("name") or "").lower()
        if any(marker in name for marker in _UNIFIED_GPU_MARKERS):
            return True
    return False


def build_profile(
    gpus: list[dict[str, Any]],
    ram_gb: float | None,
    cpu_info: dict[str, Any],
    nvme_disks: list[dict[str, Any]],
    ssd_measured: dict[str, Any],
    os_version: str | None,
    kernel: str | None,
    wsl2: bool = False,
    unified: bool = False,
    pcie_measured: dict[str, Any] | None = None,
    machine: str | None = None,
) -> dict[str, Any]:
    """Assemble an hwprofile/v1 dict from parsed collector outputs (pure, testable)."""
    nodes: list[dict[str, Any]] = [
        {
            "id": "cpu0",
            "kind": "compute",
            "class": "cpu",
            "vendor": None,
            "arch": machine,
            "model": cpu_info.get("model"),
            "cores": cpu_info.get("cores"),
            "numa_nodes": cpu_info.get("numa_nodes"),
        },
    ]
    links: list[dict[str, Any]] = []

    if unified:
        host_mem_id = "unified0"
        nodes.append(
            {
                "id": host_mem_id,
                "kind": "memory",
                "class": "unified",
                "capacity_gb": ram_gb,
                "bandwidth_gbps": {"rated": None, "measured": None},
                "pinnable": "unknown",
            }
        )
    else:
        host_mem_id = "sysram0"
        nodes.append(
            {
                "id": host_mem_id,
                "kind": "memory",
                "class": "system",
                "capacity_gb": ram_gb,
                "bandwidth_gbps": {"rated": None, "measured": None},
                "pinnable": True,
            }
        )

    gpu_label = "nogpu"
    first_gpu_mem_id: str | None = None
    for i, gpu in enumerate(gpus):
        gpu_id = f"gpu{i}"
        nodes.append(
            {
                "id": gpu_id,
                "kind": "compute",
                "class": "gpu",
                "vendor": "nvidia",
                "arch": None,
                "model": gpu.get("name"),
                "api": {
                    "cuda": gpu.get("driver_version") and "detected" or None,
                    "sm": gpu.get("compute_cap"),
                },
                "rated": {"fp16_tflops": None},
            }
        )
        if unified:
            # Coherent memory: the GPU shares the unified node; no separate vram.
            links.append(
                {
                    "from": host_mem_id,
                    "to": gpu_id,
                    "via": "unified",
                    "available": True,
                    "measured": {
                        "h2d_gbps": (pcie_measured or {}).get("h2d_gbps"),
                        "d2h_gbps": (pcie_measured or {}).get("d2h_gbps"),
                    },
                }
            )
            if i == 0:
                first_gpu_mem_id = host_mem_id
        else:
            vram_id = f"vram{i}"
            nodes.append(
                {
                    "id": vram_id,
                    "kind": "memory",
                    "class": "device_local",
                    "capacity_gb": gpu.get("vram_gb"),
                    "bandwidth_gbps": {"rated": None, "measured": None},
                    "attached_to": gpu_id,
                }
            )
            measured = {
                "h2d_gbps": (pcie_measured or {}).get("h2d_gbps") if i == 0 else None,
                "d2h_gbps": (pcie_measured or {}).get("d2h_gbps") if i == 0 else None,
                "pinned": (pcie_measured or {}).get("pinned", "unknown")
                if i == 0
                else "unknown",
            }
            links.append(
                {
                    "from": host_mem_id,
                    "to": vram_id,
                    "via": "pcie",
                    "gen": None,
                    "lanes": None,
                    "measured": measured,
                }
            )
            if i == 0:
                first_gpu_mem_id = vram_id
        if i == 0 and gpu.get("name"):
            gpu_label = sanitize_id(str(gpu["name"]))

    for j, disk in enumerate(nvme_disks):
        nvme_id = f"nvme{j}"
        nodes.append(
            {
                "id": nvme_id,
                "kind": "storage",
                "class": "nvme",
                "capacity_gb": disk.get("capacity_gb"),
                "pcie": {"gen": None, "lanes": None},
                "measured": ssd_measured if j == 0 else dict(_NULL_SSD_MEASURED),
            }
        )
        links.append(
            {
                "from": nvme_id,
                "to": host_mem_id,
                "via": "pcie",
                "measured": {
                    "seq_read_gbps": ssd_measured.get("seq_read_gbps") if j == 0 else None
                },
            }
        )
        if first_gpu_mem_id and not unified:
            links.append(
                {
                    "from": nvme_id,
                    "to": first_gpu_mem_id,
                    "via": "gds",
                    # GPUDirect Storage is not available under WSL2.
                    "available": False if wsl2 else "unknown",
                }
            )

    ram_label = f"{int(ram_gb)}ram" if ram_gb else "unknownram"
    suffix = "wsl2_detected" if wsl2 else "detected"
    profile_id = f"{gpu_label}_{ram_label}_{suffix}"

    return {
        "schema_version": "hwprofile/v1",
        "profile_id": sanitize_id(profile_id),
        "provenance": {
            "method": "detect",
            "detected_at": utc_now_iso(),
            "tool_version": "frontier-detect 0.1",
            "virtualization": "wsl2" if wsl2 else None,
            "os": {"family": "linux", "version": os_version, "kernel": kernel},
        },
        "nodes": nodes,
        "links": links,
        "envelope": {
            "power_w": {"rated": None, "measured_idle": None, "measured_load": None},
            "thermal_headroom": "unknown",
        },
    }


def detect(run_disk_bench: bool = True) -> dict[str, Any]:
    """Detect this Linux machine and return an hwprofile/v1 dict."""
    gpus = parse_nvidia_smi(run_command(_NVIDIA_SMI_QUERY))
    machine = platform.machine() or None
    wsl2 = is_wsl2()
    unified = is_unified_memory_system(gpus, machine or "")

    def _read(path: str) -> str | None:
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None

    ssd_measured: dict[str, Any] = dict(_NULL_SSD_MEASURED)
    if run_disk_bench:
        ssd_measured = disk_read_bench()

    cpu_info = parse_lscpu(run_command(["lscpu"]))
    if cpu_info.get("cores") is None:
        # Minimal containers/images lack lscpu; the kernel still knows.
        cpu_info["cores"] = os.cpu_count()

    return build_profile(
        gpus=gpus,
        ram_gb=parse_meminfo(_read("/proc/meminfo")),
        cpu_info=cpu_info,
        nvme_disks=parse_lsblk(run_command(["lsblk", "-dn", "-o", "NAME,TYPE,SIZE,TRAN", "-b"])),
        ssd_measured=ssd_measured,
        os_version=parse_os_release(_read("/etc/os-release")),
        kernel=platform.release() or None,
        wsl2=wsl2,
        unified=unified,
        pcie_measured=probe_pcie_bandwidth() if gpus else None,
        machine=machine,
    )
