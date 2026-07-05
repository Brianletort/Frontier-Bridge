"""Shared helpers for hardware detection."""

from __future__ import annotations

import os
import re
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
