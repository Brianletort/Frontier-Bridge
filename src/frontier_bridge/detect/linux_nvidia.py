"""Linux / NVIDIA detection.

Collectors: nvidia-smi, /proc/meminfo, lscpu, lsblk, and a bounded uncached
read benchmark. The PCIe H2D/D2H probe (pinned cudaMemcpy) and GPUDirect
Storage detection are future work: those links ship with measured nulls and
`available: unknown` rather than guesses.

Untested caveat: this path was written on macOS against fixture outputs. It is
fixture-tested (see tests/) but has not yet run on real Linux/NVIDIA hardware.
"""

from __future__ import annotations

import platform
from typing import Any

from frontier_bridge.detect.common import (
    bounded_disk_read_bench,
    run_command,
    sanitize_id,
    utc_now_iso,
)

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


def build_profile(
    gpus: list[dict[str, Any]],
    ram_gb: float | None,
    cpu_info: dict[str, Any],
    nvme_disks: list[dict[str, Any]],
    ssd_measured: dict[str, Any],
    os_version: str | None,
    kernel: str | None,
) -> dict[str, Any]:
    """Assemble an hwprofile/v1 dict from parsed collector outputs (pure, testable)."""
    nodes: list[dict[str, Any]] = [
        {
            "id": "cpu0",
            "kind": "compute",
            "class": "cpu",
            "vendor": None,
            "arch": None,
            "model": cpu_info.get("model"),
            "cores": cpu_info.get("cores"),
            "numa_nodes": cpu_info.get("numa_nodes"),
        },
        {
            "id": "sysram0",
            "kind": "memory",
            "class": "system",
            "capacity_gb": ram_gb,
            "bandwidth_gbps": {"rated": None, "measured": None},
            "pinnable": True,
        },
    ]
    links: list[dict[str, Any]] = []

    gpu_label = "nogpu"
    for i, gpu in enumerate(gpus):
        gpu_id, vram_id = f"gpu{i}", f"vram{i}"
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
        # PCIe probe not implemented yet: nulls, never guesses.
        links.append(
            {
                "from": "sysram0",
                "to": vram_id,
                "via": "pcie",
                "gen": None,
                "lanes": None,
                "measured": {"h2d_gbps": None, "d2h_gbps": None, "pinned": "unknown"},
            }
        )
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
                "measured": ssd_measured if j == 0 else {
                    "seq_read_gbps": None,
                    "rand_read_4k_iops": None,
                    "qd_used": None,
                    "bench_tool": None,
                },
            }
        )
        links.append(
            {
                "from": nvme_id,
                "to": "sysram0",
                "via": "pcie",
                "measured": {
                    "seq_read_gbps": ssd_measured.get("seq_read_gbps") if j == 0 else None
                },
            }
        )
        if gpus:
            links.append({"from": nvme_id, "to": "vram0", "via": "gds", "available": "unknown"})

    ram_label = f"{int(ram_gb)}ram" if ram_gb else "unknownram"
    profile_id = f"{gpu_label}_{ram_label}_detected"

    return {
        "schema_version": "hwprofile/v1",
        "profile_id": sanitize_id(profile_id),
        "provenance": {
            "method": "detect",
            "detected_at": utc_now_iso(),
            "tool_version": "frontier-detect 0.1",
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

    meminfo_text: str | None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            meminfo_text = f.read()
    except OSError:
        meminfo_text = None

    ssd_measured: dict[str, Any] = {
        "seq_read_gbps": None,
        "rand_read_4k_iops": None,
        "qd_used": None,
        "bench_tool": None,
    }
    if run_disk_bench:
        ssd_measured = bounded_disk_read_bench()

    return build_profile(
        gpus=gpus,
        ram_gb=parse_meminfo(meminfo_text),
        cpu_info=parse_lscpu(run_command(["lscpu"])),
        nvme_disks=parse_lsblk(run_command(["lsblk", "-dn", "-o", "NAME,TYPE,SIZE,TRAN", "-b"])),
        ssd_measured=ssd_measured,
        os_version=None,
        kernel=platform.release() or None,
    )
