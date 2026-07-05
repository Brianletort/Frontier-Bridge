"""Telemetry collectors for benchmark runs.

Each collector samples in a background thread and reports peaks/totals.
Everything degrades gracefully: a collector that cannot run on this system
reports nulls, never fake numbers. Built once here, reused by bench and (later)
the streaming work.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from typing import Any

from frontier_bridge.detect.common import run_command

_INTERVAL_S = 1.0


class _Sampler:
    """Background sampling loop; subclasses implement sample() and report()."""

    def __init__(self, interval_s: float = _INTERVAL_S):
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample()
            except Exception:  # noqa: BLE001 - telemetry must never kill the bench
                pass
            self._stop.wait(self._interval)

    def sample(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def report(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError


class NvidiaSmiCollector(_Sampler):
    """Peak VRAM via nvidia-smi polling (Linux/WSL2)."""

    def __init__(self) -> None:
        super().__init__()
        self.peak_vram_mib: float | None = None
        self.available = run_command(["nvidia-smi", "-L"]) is not None

    def sample(self) -> None:
        if not self.available:
            return
        out = run_command(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=10.0,
        )
        if not out:
            return
        used = sum(float(line) for line in out.split() if line.replace(".", "").isdigit())
        if used and (self.peak_vram_mib is None or used > self.peak_vram_mib):
            self.peak_vram_mib = used

    def report(self) -> dict[str, Any]:
        return {
            "peak_vram_gb": (
                round(self.peak_vram_mib * 1024 * 1024 / 1e9, 2)
                if self.peak_vram_mib
                else None
            )
        }


class ProcessMemoryCollector(_Sampler):
    """Peak RSS of the runtime process tree via `ps` (portable, no psutil dep)."""

    def __init__(self, pid: int | None) -> None:
        super().__init__()
        self.pid = pid
        self.peak_rss_kb: float | None = None

    def sample(self) -> None:
        if self.pid is None:
            return
        out = run_command(["ps", "-o", "rss=", "-p", str(self.pid)], timeout=10.0)
        if not out or not out.strip():
            return
        rss = float(out.strip().split()[0])
        if self.peak_rss_kb is None or rss > self.peak_rss_kb:
            self.peak_rss_kb = rss

    def report(self) -> dict[str, Any]:
        return {
            "peak_ram_gb": (
                round(self.peak_rss_kb * 1024 / 1e9, 2) if self.peak_rss_kb else None
            )
        }


class DiskReadCollector:
    """Total bytes read during the run (endurance tracking).

    Linux: sums sector deltas from /proc/diskstats for nvme/sd devices.
    macOS: parses cumulative bytes from `iostat -Id` style counters if
    available; otherwise reports null.
    """

    def __init__(self) -> None:
        self._start_bytes = self._read_total_bytes()

    @staticmethod
    def _read_total_bytes() -> int | None:
        # Linux path.
        try:
            with open("/proc/diskstats", encoding="utf-8") as f:
                total_sectors = 0
                for line in f:
                    fields = line.split()
                    if len(fields) < 6:
                        continue
                    device = fields[2]
                    if re.fullmatch(r"(nvme\d+n\d+|sd[a-z]+)", device):
                        total_sectors += int(fields[5])  # sectors read
                return total_sectors * 512
        except OSError:
            pass
        # macOS path: cumulative MB transferred per disk.
        out = run_command(["iostat", "-Id"], timeout=10.0)
        if out:
            try:
                lines = [l for l in out.strip().splitlines() if l.strip()]
                values = lines[-1].split()
                # Columns repeat (KB/t, xfrs, MB) per disk; MB is every 3rd.
                mbs = [float(values[i]) for i in range(2, len(values), 3)]
                return int(sum(mbs) * 1e6)
            except (ValueError, IndexError):
                return None
        return None

    def report(self) -> dict[str, Any]:
        end_bytes = self._read_total_bytes()
        if self._start_bytes is None or end_bytes is None:
            return {"ssd_total_read_gb": None}
        return {"ssd_total_read_gb": round((end_bytes - self._start_bytes) / 1e9, 2)}


class PowerMetricsCollector(_Sampler):
    """macOS package power via powermetrics. Needs sudo; degrades to nulls."""

    def __init__(self) -> None:
        super().__init__(interval_s=5.0)
        self.samples_w: list[float] = []
        probe = run_command(
            ["sudo", "-n", "powermetrics", "-n", "1", "-i", "100", "--samplers", "cpu_power"],
            timeout=15.0,
        )
        self.available = bool(probe and "Power" in probe)

    def sample(self) -> None:
        if not self.available:
            return
        out = run_command(
            ["sudo", "-n", "powermetrics", "-n", "1", "-i", "100", "--samplers", "cpu_power"],
            timeout=15.0,
        )
        if not out:
            return
        match = re.search(r"Combined Power.*?:\s*(\d+)\s*mW", out)
        if match:
            self.samples_w.append(int(match.group(1)) / 1000)

    def report(self) -> dict[str, Any]:
        if not self.samples_w:
            return {"power_w_avg": None}
        return {"power_w_avg": round(sum(self.samples_w) / len(self.samples_w), 1)}


def start_collectors(runtime_pid: int | None) -> tuple[list[Any], DiskReadCollector]:
    """Start all applicable samplers; returns (samplers, disk_counter)."""
    samplers: list[Any] = [
        NvidiaSmiCollector(),
        ProcessMemoryCollector(runtime_pid),
        PowerMetricsCollector(),
    ]
    for sampler in samplers:
        sampler.start()
    return samplers, DiskReadCollector()


def stop_and_report(samplers: list[Any], disk: DiskReadCollector) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for sampler in samplers:
        sampler.stop()
        report.update(sampler.report())
    report.update(disk.report())
    return report
