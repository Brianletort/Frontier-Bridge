"""HF GGUF repo ingestion: repo parsing, quant grouping, profile assembly.

All network access is faked; the emitted profile must validate against
modelprofile/v1 and carry LFS sha256 pins, measured sizes, and nulls (never
guesses) for anything unmeasured.
"""

import pytest

from frontier_bridge.ingest import (
    IngestError,
    build_profile,
    group_quants,
    ingest_repo,
    parse_repo,
    select_quant,
    shard_urls,
)
from frontier_bridge.validation import validate_instance

_SHA_A = "a" * 64
_SHA_B = "b" * 64


def _tree_entry(path: str, size: int, sha: str | None = _SHA_A) -> dict:
    entry = {"type": "file", "path": path, "size": size}
    if sha:
        entry["lfs"] = {"oid": sha, "size": size}
    return entry


_TREE = [
    _tree_entry("Q2_K_XL/Model-Q2_K_XL-00001-of-00002.gguf", 50_000_000_000, _SHA_A),
    _tree_entry("Q2_K_XL/Model-Q2_K_XL-00002-of-00002.gguf", 57_000_000_000, _SHA_B),
    _tree_entry("Q4_K_M/Model-Q4_K_M-00001-of-00001.gguf", 200_000_000_000, _SHA_A),
    _tree_entry("README.md", 1000, None),
]


def _fake_fetch(url: str):
    if "/tree/" in url:
        return _TREE
    return {"cardData": {"license": "mit"}}


def test_parse_repo_variants():
    assert parse_repo("org/name") == ("org/name", "main")
    assert parse_repo("https://huggingface.co/org/name") == ("org/name", "main")
    assert parse_repo("https://huggingface.co/org/name/tree/main/Q2_K_XL") == (
        "org/name",
        "main",
    )
    with pytest.raises(IngestError):
        parse_repo("just-a-name")


def test_group_quants_by_directory():
    groups = group_quants(_TREE)
    assert [g.upstream_name for g in groups] == ["Q2_K_XL", "Q4_K_M"]
    q2 = groups[0]
    assert len(q2.files) == 2
    assert q2.quant_id == "q2_k_xl"
    assert q2.size_gb == 107.0


def test_group_quants_flat_files_by_token():
    tree = [
        _tree_entry("Model-UD-Q4_K_XL-00001-of-00002.gguf", 10),
        _tree_entry("Model-UD-Q4_K_XL-00002-of-00002.gguf", 10),
        _tree_entry("Model-IQ2_XS.gguf", 5),
    ]
    groups = group_quants(tree)
    names = {g.upstream_name: len(g.files) for g in groups}
    assert names == {"UD-Q4_K_XL": 2, "IQ2_XS": 1}


def test_select_quant():
    groups = group_quants(_TREE)
    assert select_quant(groups, "Q2_K_XL").upstream_name == "Q2_K_XL"
    assert select_quant(groups, "q4_k_m").upstream_name == "Q4_K_M"
    with pytest.raises(IngestError, match="multiple quants"):
        select_quant(groups, None)
    with pytest.raises(IngestError, match="not found"):
        select_quant(groups, "q8_0")


def test_shard_urls():
    group = select_quant(group_quants(_TREE), "Q2_K_XL")
    urls = shard_urls("org/name", "main", group)
    assert urls[0] == (
        "https://huggingface.co/org/name/resolve/main/"
        "Q2_K_XL/Model-Q2_K_XL-00001-of-00002.gguf"
    )


def test_build_profile_without_inspection_is_null_not_guessed():
    group = select_quant(group_quants(_TREE), "Q2_K_XL")
    profile = build_profile("org/name", "main", group, model_id="model-x")
    assert validate_instance(profile) == []
    assert profile["architecture"]["params_total_b"] is None
    assert profile["memory_model"]["dense_resident_gb"]["q2_k_xl"] is None
    artifact = profile["artifacts"][0]
    assert artifact["size_gb"] == 107.0
    assert artifact["sha256"] == _SHA_A
    assert [s["sha256"] for s in artifact["shards"]] == [_SHA_A, _SHA_B]


def test_build_profile_with_inspection_fills_measured_fields():
    group = select_quant(group_quants(_TREE), "Q2_K_XL")
    inspection = {
        "expert_count": 256,
        "expert_used_count": 8,
        "expert_shared_count": 1,
        "context_length": 131072,
        "params_total_b": 284.0,
        "dense_resident_gb": 6.75,
        "per_expert_layer_gb": 0.0091,
        "routed_experts_gb": 100.28,
        "method": "gguf_header_offset_deltas",
    }
    profile = build_profile(
        "org/name", "main", group, model_id="model-x", inspection=inspection
    )
    assert validate_instance(profile) == []
    arch = profile["architecture"]
    assert arch["type"] == "moe"
    assert arch["params_total_b"] == 284.0
    assert arch["experts"]["routed_total"] == 256
    assert arch["context_max"] == 131072
    mm = profile["memory_model"]
    assert mm["dense_resident_gb"]["q2_k_xl"] == 6.75
    assert mm["measurement"]["routed_experts_gb"]["q2_k_xl"] == 100.28


def test_ingest_repo_end_to_end_with_fake_fetch():
    profile = ingest_repo(
        "https://huggingface.co/org/Some-Model-GGUF",
        quant="Q2_K_XL",
        inspect_headers=False,
        fetch=_fake_fetch,
    )
    assert validate_instance(profile) == []
    assert profile["model_id"] == "some-model"
    assert profile["license"]["name"] == "mit"
    assert profile["runtime_support"][0]["runtime"] == "llama_cpp"
