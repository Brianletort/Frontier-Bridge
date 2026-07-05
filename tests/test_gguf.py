"""GGUF header inspection tested against a synthetic GGUF file built in-test,
plus a live header check against the pinned GLM-5.2 shard when the network
allows (skipped offline)."""

import struct
import urllib.error

import pytest

from frontier_bridge.gguf import (
    GGUFError,
    classify_tensor,
    inspect_artifact,
    inspect_shard,
)

_TYPE_U32, _TYPE_U64, _TYPE_STR = 4, 10, 8
_ALIGNMENT = 32


def _kv_str(key: str, value: str) -> bytes:
    key_b, val_b = key.encode(), value.encode()
    return (
        struct.pack("<Q", len(key_b)) + key_b
        + struct.pack("<I", _TYPE_STR)
        + struct.pack("<Q", len(val_b)) + val_b
    )


def _kv_u32(key: str, value: int) -> bytes:
    key_b = key.encode()
    return (
        struct.pack("<Q", len(key_b)) + key_b
        + struct.pack("<I", _TYPE_U32)
        + struct.pack("<I", value)
    )


def _tensor_info(name: str, dims: list[int], offset: int) -> bytes:
    name_b = name.encode()
    out = struct.pack("<Q", len(name_b)) + name_b
    out += struct.pack("<I", len(dims))
    for d in dims:
        out += struct.pack("<Q", d)
    out += struct.pack("<I", 0)  # ggml type f32 (irrelevant: sizes come from offsets)
    out += struct.pack("<Q", offset)
    return out


def make_gguf(path, tensors: list[tuple[str, int]], metadata: dict[str, int] | None = None):
    """Write a minimal valid GGUF v3 file with given (name, data_size) tensors."""
    kvs = [_kv_str("general.architecture", "testarch")]
    for key, value in (metadata or {}).items():
        kvs.append(_kv_u32(key, value))
    kv_blob = b"".join(kvs)

    infos = b""
    offset = 0
    for name, size in tensors:
        infos += _tensor_info(name, [size // 4], offset)
        offset += size  # sizes chosen alignment-friendly in tests

    header = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(kvs)) + kv_blob + infos
    padding = (-len(header)) % _ALIGNMENT
    data = b"\x00" * offset
    path.write_bytes(header + b"\x00" * padding + data)


def test_classify_tensor():
    assert classify_tensor("blk.5.ffn_gate_exps.weight") == "routed_experts"
    assert classify_tensor("blk.5.ffn_gate_shexp.weight") == "shared_experts"
    assert classify_tensor("blk.5.attn_q.weight") == "dense"
    assert classify_tensor("token_embd.weight") == "dense"


def test_inspect_synthetic_shard(tmp_path):
    gguf_path = tmp_path / "model.gguf"
    tensors = [
        ("token_embd.weight", 1024),
        ("blk.0.attn_q.weight", 512),
        ("blk.0.ffn_gate_exps.weight", 4096),
        ("blk.0.ffn_gate_shexp.weight", 256),
    ]
    make_gguf(
        gguf_path,
        tensors,
        metadata={"testarch.expert_count": 4, "testarch.block_count": 1},
    )
    shard = inspect_shard(str(gguf_path))
    assert shard.arch == "testarch"
    sizes = {t.name: t.size_bytes for t in shard.tensors}
    assert sizes["token_embd.weight"] == 1024
    assert sizes["blk.0.ffn_gate_exps.weight"] == 4096
    assert sizes["blk.0.ffn_gate_shexp.weight"] == 256


def test_inspect_artifact_summary(tmp_path):
    gguf_path = tmp_path / "model.gguf"
    make_gguf(
        gguf_path,
        [
            ("token_embd.weight", 1_000_000),
            ("blk.0.ffn_gate_exps.weight", 8_000_000),
            ("blk.0.ffn_up_shexp.weight", 500_000),
        ],
        metadata={"testarch.expert_count": 8, "testarch.block_count": 1},
    )
    summary = inspect_artifact([str(gguf_path)])
    assert summary["dense_resident_gb"] == round(1_500_000 / 1e9, 2)
    assert summary["routed_experts_gb"] == round(8_000_000 / 1e9, 2)
    assert summary["per_expert_layer_gb"] == round(8_000_000 / 8 / 1e9, 4)
    assert summary["method"] == "gguf_header_offset_deltas"


def test_bad_magic_raises(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\x00" * 100)
    with pytest.raises(GGUFError, match="bad magic"):
        inspect_shard(str(bad))


def test_live_glm_shard_header():
    """Range-request the first pinned GLM-5.2 Q4 shard header. Network-gated."""
    url = (
        "https://huggingface.co/unsloth/GLM-5.2-GGUF/resolve/main/"
        "UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf"
    )
    try:
        shard = inspect_shard(url)
    except (urllib.error.URLError, OSError, TimeoutError):
        pytest.skip("network unavailable")
    assert shard.arch is not None
    assert shard.file_size > 0
