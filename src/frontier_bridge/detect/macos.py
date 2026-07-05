"""macOS / Apple Silicon detection.

Collectors: sysctl, system_profiler, shutil.disk_usage, and a bounded uncached
read benchmark. powermetrics needs sudo, so the power envelope degrades
gracefully to nulls.
"""

from __future__ import annotations

import json
import platform
import shutil
from typing import Any

from frontier_bridge.detect.common import (
    disk_read_bench,
    run_command,
    sanitize_id,
    utc_now_iso,
)


def _sysctl(name: str) -> str | None:
    out = run_command(["sysctl", "-n", name])
    return out.strip() if out else None


def parse_hardware_overview(system_profiler_json: str | None) -> dict[str, Any]:
    """Extract chip name and core counts from `system_profiler SPHardwareDataType -json`."""
    info: dict[str, Any] = {"chip": None, "model": None, "cores_description": None}
    if not system_profiler_json:
        return info
    try:
        data = json.loads(system_profiler_json)
        hw = data["SPHardwareDataType"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return info
    info["chip"] = hw.get("chip_type")
    info["model"] = hw.get("machine_model")
    info["cores_description"] = hw.get("number_processors")
    return info


def detect(run_disk_bench: bool = True) -> dict[str, Any]:
    """Detect this Mac and return an hwprofile/v1 dict."""
    mem_bytes = _sysctl("hw.memsize")
    ncpu = _sysctl("hw.ncpu")
    brand = _sysctl("machdep.cpu.brand_string")
    overview = parse_hardware_overview(
        run_command(["system_profiler", "SPHardwareDataType", "-json"])
    )
    chip = overview["chip"] or brand

    capacity_gb = round(int(mem_bytes) / 1e9, 1) if mem_bytes else None
    cores = int(ncpu) if ncpu else None

    disk_total_gb: float | None
    try:
        disk_total_gb = round(shutil.disk_usage("/").total / 1e9, 1)
    except OSError:
        disk_total_gb = None

    ssd_measured: dict[str, Any] = {
        "seq_read_gbps": None,
        "rand_read_4k_iops": None,
        "qd_used": None,
        "bench_tool": None,
    }
    if run_disk_bench:
        ssd_measured = disk_read_bench()

    chip_id = sanitize_id(chip or "apple_silicon")
    mem_label = f"{int(capacity_gb)}gb" if capacity_gb else "unknownram"
    profile_id = f"{chip_id}_{mem_label}_detected"

    return {
        "schema_version": "hwprofile/v1",
        "profile_id": profile_id,
        "provenance": {
            "method": "detect",
            "detected_at": utc_now_iso(),
            "tool_version": "frontier-detect 0.1",
            "os": {
                "family": "macos",
                "version": platform.mac_ver()[0] or None,
                "kernel": platform.release() or None,
            },
        },
        "nodes": [
            {
                "id": "cpu0",
                "kind": "compute",
                "class": "cpu",
                "vendor": "apple",
                "arch": "arm64",
                "model": chip,
                "cores": cores,
                "numa_nodes": 1,
            },
            {
                "id": "gpu0",
                "kind": "compute",
                "class": "gpu",
                "vendor": "apple",
                "arch": chip_id,
                "model": chip,
                "api": {"metal": "detected"},
                "rated": {"fp16_tflops": None},
            },
            {
                "id": "unified0",
                "kind": "memory",
                "class": "unified",
                "capacity_gb": capacity_gb,
                "bandwidth_gbps": {"rated": None, "measured": None},
                "pinnable": "unknown",
            },
            {
                "id": "ssd0",
                "kind": "storage",
                "class": "internal_ssd",
                "capacity_gb": disk_total_gb,
                "measured": ssd_measured,
            },
        ],
        "links": [
            {
                "from": "unified0",
                "to": "gpu0",
                "via": "unified",
                "available": True,
                "measured": {"h2d_gbps": None, "d2h_gbps": None},
            },
            {
                "from": "ssd0",
                "to": "unified0",
                "via": "internal",
                "measured": {"seq_read_gbps": ssd_measured.get("seq_read_gbps")},
            },
        ],
        "envelope": {
            "power_w": {"rated": None, "measured_idle": None, "measured_load": None},
            "thermal_headroom": "unknown",
        },
    }
