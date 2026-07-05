"""Shared helpers for hardware detection."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

# macOS fcntl constant to bypass the buffer cache for a file descriptor.
_F_NOCACHE_DARWIN = 48

_BENCH_FILE_MB = 512
_BENCH_CHUNK_MB = 8


def run_command(args: list[str], timeout: float = 30.0) -> str | None:
    """Run a command and return stdout, or None if it fails or is missing."""
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize_id(raw: str) -> str:
    """Lowercase and reduce to [a-z0-9_] for profile/node ids."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    return cleaned or "unknown"


def is_wsl2(proc_version_text: str | None = None) -> bool:
    """Detect WSL2 from /proc/version content or the WSL_DISTRO_NAME env var."""
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    text = proc_version_text
    if text is None:
        try:
            with open("/proc/version", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            return False
    return "microsoft" in text.lower()


def parse_fio_json(output: str | None) -> dict[str, Any] | None:
    """Extract a 'measured' block from `fio --output-format=json` output."""
    if not output:
        return None
    try:
        data = json.loads(output)
        job = data["jobs"][0]
        read = job["read"]
        bw_bytes = read.get("bw_bytes")
        iops = read.get("iops")
        iodepth = (job.get("job options") or {}).get("iodepth")
        fio_version = data.get("fio version")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None
    if not bw_bytes:
        return None
    return {
        "seq_read_gbps": round(bw_bytes / 1e9, 2),
        "rand_read_4k_iops": round(iops) if iops else None,
        "qd_used": int(iodepth) if iodepth else None,
        "bench_tool": f"{fio_version or 'fio'} seq read 1GB direct=1",
    }


def fio_disk_read_bench(directory: str | None = None) -> dict[str, Any] | None:
    """Run a bounded fio sequential-read benchmark if fio is installed.

    Uses direct I/O (no page cache) and JSON output. Returns None when fio is
    missing or fails, so callers can fall back to the Python bench.
    """
    if shutil.which("fio") is None:
        return None
    bench_dir = directory or tempfile.gettempdir()
    ioengine = "posixaio" if sys.platform == "darwin" else "libaio"
    output = run_command(
        [
            "fio",
            "--name=frontier_seq_read",
            "--rw=read",
            "--bs=1m",
            "--size=1g",
            "--iodepth=32",
            f"--ioengine={ioengine}",
            "--direct=1",
            "--unlink=1",
            f"--directory={bench_dir}",
            "--output-format=json",
        ],
        timeout=120.0,
    )
    return parse_fio_json(output)


def disk_read_bench(directory: str | None = None) -> dict[str, Any]:
    """Preferred disk bench: fio when installed, Python bounded read otherwise."""
    fio_result = fio_disk_read_bench(directory)
    if fio_result is not None:
        return fio_result
    return bounded_disk_read_bench(directory)


def parse_nvbandwidth(output: str | None) -> dict[str, Any] | None:
    """Extract H2D/D2H GB/s from `nvbandwidth` SUM lines.

    Expected lines look like: ``SUM host_to_device_memcpy_ce 55.23``.
    Returns None if neither direction is found.
    """
    if not output:
        return None
    h2d = d2h = None
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 3 or fields[0] != "SUM":
            continue
        try:
            value = round(float(fields[-1]), 1)
        except ValueError:
            continue
        if "host_to_device" in fields[1]:
            h2d = value
        elif "device_to_host" in fields[1]:
            d2h = value
    if h2d is None and d2h is None:
        return None
    return {
        "h2d_gbps": h2d,
        "d2h_gbps": d2h,
        "pinned": True,
        "bench_tool": "nvbandwidth host/device memcpy CE",
    }


def _nvbandwidth_probe() -> dict[str, Any] | None:
    if shutil.which("nvbandwidth") is None:
        return None
    output = run_command(
        [
            "nvbandwidth",
            "-t",
            "host_to_device_memcpy_ce",
            "device_to_host_memcpy_ce",
        ],
        timeout=120.0,
    )
    return parse_nvbandwidth(output)


def probe_pcie_bandwidth() -> dict[str, Any] | None:
    """Measure pinned H2D/D2H bandwidth: nvbandwidth if on PATH, else a
    cuda-python pinned memcpy, else None (caller records nulls, never guesses).

    Written against current tool/API docs; verify on real hardware — this path
    has not run on a CUDA machine yet.
    """
    nvb = _nvbandwidth_probe()
    if nvb is not None:
        return nvb
    return _cuda_python_probe()


def _cuda_python_probe() -> dict[str, Any] | None:
    try:
        from cuda import cudart
    except ImportError:
        return None

    size = 256 * 1024 * 1024  # 256 MB per direction
    repeats = 4
    try:
        err, dev_ptr = cudart.cudaMalloc(size)
        if err != cudart.cudaError_t.cudaSuccess:
            return None
        err, host_ptr = cudart.cudaMallocHost(size)
        if err != cudart.cudaError_t.cudaSuccess:
            cudart.cudaFree(dev_ptr)
            return None

        def _timed_copy(dst: int, src: int, kind: Any) -> float | None:
            start = time.perf_counter()
            for _ in range(repeats):
                (copy_err,) = cudart.cudaMemcpy(dst, src, size, kind)
                if copy_err != cudart.cudaError_t.cudaSuccess:
                    return None
            (sync_err,) = cudart.cudaDeviceSynchronize()
            if sync_err != cudart.cudaError_t.cudaSuccess:
                return None
            elapsed = time.perf_counter() - start
            return round(size * repeats / elapsed / 1e9, 1)

        h2d = _timed_copy(
            dev_ptr, host_ptr, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
        d2h = _timed_copy(
            host_ptr, dev_ptr, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
        cudart.cudaFreeHost(host_ptr)
        cudart.cudaFree(dev_ptr)
    except Exception:  # noqa: BLE001 - any CUDA failure means "unknown", not a crash
        return None
    if h2d is None and d2h is None:
        return None
    return {
        "h2d_gbps": h2d,
        "d2h_gbps": d2h,
        "pinned": True,
        "bench_tool": f"cuda-python pinned memcpy {size // (1024 * 1024)}MB x{repeats}",
    }


def bounded_disk_read_bench(directory: str | None = None) -> dict[str, Any]:
    """Measure sequential read throughput with a bounded (512 MB) uncached read.

    Writes a scratch file, drops it from the page cache (F_NOCACHE on macOS,
    posix_fadvise(DONTNEED) on Linux), then times a sequential read. Returns a
    'measured' block for a storage node; values are null if the bench fails.
    """
    result: dict[str, Any] = {
        "seq_read_gbps": None,
        "rand_read_4k_iops": None,
        "qd_used": 1,
        "bench_tool": f"frontier-detect python bounded read ({_BENCH_FILE_MB}MB, uncached, qd1)",
    }
    chunk = os.urandom(_BENCH_CHUNK_MB * 1024 * 1024)
    n_chunks = _BENCH_FILE_MB // _BENCH_CHUNK_MB
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=True) as tmp:
            for _ in range(n_chunks):
                tmp.write(chunk)
            tmp.flush()
            os.fsync(tmp.fileno())

            fd = os.open(tmp.name, os.O_RDONLY)
            try:
                if sys.platform == "darwin":
                    import fcntl

                    fcntl.fcntl(fd, _F_NOCACHE_DARWIN, 1)
                elif hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                total = 0
                start = time.perf_counter()
                while True:
                    data = os.read(fd, _BENCH_CHUNK_MB * 1024 * 1024)
                    if not data:
                        break
                    total += len(data)
                elapsed = time.perf_counter() - start
            finally:
                os.close(fd)
        if elapsed > 0 and total > 0:
            result["seq_read_gbps"] = round(total / elapsed / 1e9, 2)
    except OSError:
        pass
    return result
