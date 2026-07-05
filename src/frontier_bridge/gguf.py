"""GGUF header inspection without full downloads.

Parses the GGUF header (metadata + tensor infos) from a local file or an HTTP
URL via range requests, then computes per-tensor byte sizes from the offset
deltas in the header — no quant-type size tables, no guessing. Classifies
tensors into routed experts / shared experts / dense using llama.cpp naming
conventions, which is what the memory planner needs:

    dense (+ shared experts + router) -> must stay resident
    routed experts                    -> tierable across VRAM / RAM / SSD

Multi-shard models: inspect each shard and sum (each shard has its own header).
"""

from __future__ import annotations

import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"
_DEFAULT_ALIGNMENT = 32
_CHUNK = 4 * 1024 * 1024

# GGUF metadata value type ids -> struct format (fixed-size scalars).
_SCALAR_FMT = {
    0: "<B",  # uint8
    1: "<b",  # int8
    2: "<H",  # uint16
    3: "<h",  # int16
    4: "<I",  # uint32
    5: "<i",  # int32
    6: "<f",  # float32
    7: "<?",  # bool
    10: "<Q",  # uint64
    11: "<q",  # int64
    12: "<d",  # float64
}
_TYPE_STRING = 8
_TYPE_ARRAY = 9

# llama.cpp tensor-name markers.
_ROUTED_MARKERS = ("_exps.",)
_SHARED_MARKERS = ("_shexp.",)


class GGUFError(Exception):
    """Malformed or unsupported GGUF header."""


class _RangeSource:
    """Random-access byte source over a local file or an HTTP(S) URL."""

    def __init__(self, location: str):
        self.location = location
        self._is_url = location.startswith(("http://", "https://"))
        self._file: BinaryIO | None = None
        self._size: int | None = None
        if not self._is_url:
            path = Path(location)
            self._file = path.open("rb")
            self._size = path.stat().st_size

    def read(self, offset: int, length: int) -> bytes:
        if self._file is not None:
            self._file.seek(offset)
            return self._file.read(length)
        request = urllib.request.Request(
            self.location,
            headers={"Range": f"bytes={offset}-{offset + length - 1}"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            content_range = response.headers.get("Content-Range")
            if self._size is None and content_range and "/" in content_range:
                total = content_range.rsplit("/", 1)[1]
                if total.isdigit():
                    self._size = int(total)
            return response.read()

    @property
    def size(self) -> int:
        if self._size is None:
            # Trigger a tiny range request to learn the total size.
            self.read(0, 1)
        if self._size is None:
            raise GGUFError(f"could not determine size of {self.location}")
        return self._size

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


class _BufferedReader:
    """Sequential reader over a _RangeSource that fetches in chunks."""

    def __init__(self, source: _RangeSource):
        self._source = source
        self._buffer = b""
        self._buffer_start = 0
        self.position = 0

    def _ensure(self, length: int) -> None:
        end = self.position + length
        buffer_end = self._buffer_start + len(self._buffer)
        if self.position >= self._buffer_start and end <= buffer_end:
            return
        fetch_len = max(length, _CHUNK)
        self._buffer = self._source.read(self.position, fetch_len)
        self._buffer_start = self.position
        if len(self._buffer) < length:
            raise GGUFError("unexpected end of file while parsing GGUF header")

    def take(self, length: int) -> bytes:
        self._ensure(length)
        start = self.position - self._buffer_start
        data = self._buffer[start : start + length]
        self.position += length
        return data

    def scalar(self, type_id: int) -> Any:
        fmt = _SCALAR_FMT[type_id]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def u32(self) -> int:
        return self.scalar(4)

    def u64(self) -> int:
        return self.scalar(10)

    def string(self) -> str:
        length = self.u64()
        if length > 64 * 1024 * 1024:
            raise GGUFError(f"implausible string length {length}")
        return self.take(length).decode("utf-8", errors="replace")

    def skip_value(self, type_id: int, capture: bool = False) -> Any:
        """Read (or skip) one metadata value. Returns it when capture=True."""
        if type_id in _SCALAR_FMT:
            value = self.scalar(type_id)
            return value if capture else None
        if type_id == _TYPE_STRING:
            value = self.string()
            return value if capture else None
        if type_id == _TYPE_ARRAY:
            element_type = self.u32()
            count = self.u64()
            if element_type in _SCALAR_FMT:
                item_size = struct.calcsize(_SCALAR_FMT[element_type])
                self.position += item_size * count  # skip without fetching
                return None
            if element_type == _TYPE_STRING:
                for _ in range(count):
                    self.string()
                return None
            raise GGUFError(f"nested arrays of type {element_type} unsupported")
        raise GGUFError(f"unknown metadata value type {type_id}")


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    ggml_type: int
    offset: int
    size_bytes: int = 0


@dataclass
class ShardInspection:
    location: str
    arch: str | None
    alignment: int
    metadata: dict[str, Any]
    tensors: list[TensorInfo]
    data_start: int
    file_size: int


# Metadata keys worth capturing (per-arch expert counts use the arch prefix).
_CAPTURE_SUFFIXES = (
    "general.architecture",
    "general.alignment",
    ".expert_count",
    ".expert_used_count",
    ".expert_shared_count",
    ".block_count",
    ".context_length",
)


def inspect_shard(location: str) -> ShardInspection:
    """Parse one GGUF shard's header and compute tensor sizes from offsets."""
    source = _RangeSource(location)
    try:
        reader = _BufferedReader(source)
        if reader.take(4) != GGUF_MAGIC:
            raise GGUFError(f"{location} is not a GGUF file (bad magic)")
        version = reader.u32()
        if version < 2:
            raise GGUFError(f"GGUF version {version} too old (need v2+)")
        tensor_count = reader.u64()
        kv_count = reader.u64()
        if tensor_count > 10_000_000 or kv_count > 1_000_000:
            raise GGUFError("implausible header counts")

        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = reader.string()
            type_id = reader.u32()
            capture = any(key.endswith(suffix) for suffix in _CAPTURE_SUFFIXES)
            value = reader.skip_value(type_id, capture=capture)
            if capture:
                metadata[key] = value

        tensors: list[TensorInfo] = []
        for _ in range(tensor_count):
            name = reader.string()
            n_dims = reader.u32()
            dims = [reader.u64() for _ in range(n_dims)]
            ggml_type = reader.u32()
            offset = reader.u64()
            tensors.append(TensorInfo(name=name, dims=dims, ggml_type=ggml_type, offset=offset))

        alignment = int(metadata.get("general.alignment", _DEFAULT_ALIGNMENT))
        header_end = reader.position
        data_start = (header_end + alignment - 1) // alignment * alignment
        file_size = source.size

        # Sizes from offset deltas: no quant-type tables, no guessing.
        by_offset = sorted(tensors, key=lambda t: t.offset)
        for current, following in zip(by_offset, by_offset[1:]):
            current.size_bytes = following.offset - current.offset
        if by_offset:
            by_offset[-1].size_bytes = (file_size - data_start) - by_offset[-1].offset

        return ShardInspection(
            location=location,
            arch=metadata.get("general.architecture"),
            alignment=alignment,
            metadata=metadata,
            tensors=tensors,
            data_start=data_start,
            file_size=file_size,
        )
    finally:
        source.close()


def classify_tensor(name: str) -> str:
    """routed_experts | shared_experts | dense, by llama.cpp naming convention."""
    if any(marker in name for marker in _ROUTED_MARKERS):
        return "routed_experts"
    if any(marker in name for marker in _SHARED_MARKERS):
        return "shared_experts"
    return "dense"


def inspect_artifact(locations: list[str]) -> dict[str, Any]:
    """Inspect one or more GGUF shards and return a memory-model summary."""
    totals = {"routed_experts": 0, "shared_experts": 0, "dense": 0}
    arch: str | None = None
    expert_count: int | None = None
    block_count: int | None = None
    tensor_count = 0

    for location in locations:
        shard = inspect_shard(location)
        arch = arch or shard.arch
        tensor_count += len(shard.tensors)
        for key, value in shard.metadata.items():
            if key.endswith(".expert_count") and isinstance(value, int):
                expert_count = value
            elif key.endswith(".block_count") and isinstance(value, int):
                block_count = value
        for tensor in shard.tensors:
            totals[classify_tensor(tensor.name)] += tensor.size_bytes

    routed_gb = round(totals["routed_experts"] / 1e9, 2)
    dense_resident_gb = round(
        (totals["dense"] + totals["shared_experts"]) / 1e9, 2
    )
    per_expert_gb = None
    if expert_count and block_count and totals["routed_experts"]:
        # Per (expert, layer) slice: the tierable streaming unit.
        per_expert_gb = round(
            totals["routed_experts"] / (expert_count * block_count) / 1e9, 4
        )

    return {
        "arch": arch,
        "shards": len(locations),
        "tensor_count": tensor_count,
        "expert_count": expert_count,
        "block_count": block_count,
        "total_gb": round(sum(totals.values()) / 1e9, 2),
        "dense_resident_gb": dense_resident_gb,
        "routed_experts_gb": routed_gb,
        "shared_experts_gb": round(totals["shared_experts"] / 1e9, 2),
        "per_expert_layer_gb": per_expert_gb,
        "method": "gguf_header_offset_deltas",
    }
