"""Expert-sized SSD read microbenchmark: the measured floor of L2 stream-on-miss.

Sequential bandwidth flatters an SSD. Stream-on-miss reads (expert, layer)
slices — megabytes at random offsets — so the planner's worst-case miss math
should use bandwidth measured at expert granularity, not `seq_read_gbps`.

Method: write a scratch file, bypass the page cache (F_NOCACHE on macOS,
posix_fadvise DONTNEED on Linux), then time uncached reads of chunk_mb-sized
blocks at random chunk-aligned offsets. One record per chunk size.
"""

from __future__ import annotations

import fcntl
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

_F_NOCACHE_DARWIN = 48
_WRITE_CHUNK = 8 * 1024 * 1024


def _uncache(fd: int) -> None:
    if sys.platform == "darwin":
        fcntl.fcntl(fd, _F_NOCACHE_DARWIN, 1)
    elif hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def expert_read_bench(
    chunk_mbs: list[int],
    file_gb: float = 4.0,
    reads_per_size: int = 48,
    directory: str | None = None,
    seed: int = 20260705,
) -> list[dict[str, Any]]:
    """Measure uncached random reads at expert-sized granularity.

    Returns one record per chunk size with GB/s, effective reads/s, and the
    resulting all-miss decode floor helper numbers left to the caller.
    """
    rng = random.Random(seed)
    file_bytes = int(file_gb * 1e9)
    records: list[dict[str, Any]] = []
    filler = os.urandom(_WRITE_CHUNK)

    with tempfile.NamedTemporaryFile(dir=directory, delete=True) as tmp:
        # Keep write pages out of the cache too — otherwise the "uncached"
        # reads get served from the pages the write just populated.
        _uncache(tmp.fileno())
        written = 0
        while written < file_bytes:
            tmp.write(filler)
            written += len(filler)
        tmp.flush()
        os.fsync(tmp.fileno())
        _uncache(tmp.fileno())

        for chunk_mb in chunk_mbs:
            chunk_bytes = chunk_mb * 1024 * 1024
            max_offset_chunks = max(written // chunk_bytes - 1, 1)
            fd = os.open(tmp.name, os.O_RDONLY)
            try:
                _uncache(fd)
                total = 0
                start = time.perf_counter()
                for _ in range(reads_per_size):
                    offset = rng.randrange(max_offset_chunks) * chunk_bytes
                    os.lseek(fd, offset, os.SEEK_SET)
                    remaining = chunk_bytes
                    while remaining > 0:
                        data = os.read(fd, min(remaining, _WRITE_CHUNK))
                        if not data:
                            break
                        remaining -= len(data)
                        total += len(data)
                elapsed = time.perf_counter() - start
            finally:
                os.close(fd)
            gbps = round(total / elapsed / 1e9, 2) if elapsed > 0 and total else None
            records.append(
                {
                    "chunk_mb": chunk_mb,
                    "reads": reads_per_size,
                    "gbps": gbps,
                    "reads_per_s": (
                        round(reads_per_size / elapsed, 1) if elapsed > 0 else None
                    ),
                    "file_gb": round(written / 1e9, 1),
                    "method": "uncached_random_chunk_reads_python",
                    "measured_at": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                }
            )
    return records


def stream_read_gbps(records: list[dict[str, Any]]) -> float | None:
    """Conservative single figure for planner streaming math: the worst
    measured expert-sized bandwidth across chunk sizes."""
    values = [r["gbps"] for r in records if r.get("gbps")]
    return min(values) if values else None
