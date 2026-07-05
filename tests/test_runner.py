"""Runner: launch-command substitution, hash verification, endpoint probing."""

import hashlib

import pytest

from frontier_bridge.runner import (
    RunError,
    build_launch,
    sha256_file,
    verify_artifact,
)


def _plan(launch: str) -> dict:
    return {
        "schema_version": "plan/v1",
        "runtime": {"engine": "llama_cpp", "launch": launch},
    }


def test_build_launch_substitutes_path_and_finds_port():
    plan = _plan(
        "llama-server -m <GGUF_PATH> -c 32768 --host 127.0.0.1 --port 8123 -ngl 999  # comment"
    )
    spec = build_launch(plan, "/models/glm.gguf")
    assert "<GGUF_PATH>" not in spec.command
    assert "/models/glm.gguf" in spec.args
    assert spec.port == 8123
    assert "#" not in spec.command  # advisory comment stripped before exec


def test_build_launch_refuses_empty_launch():
    with pytest.raises(RunError, match="no runtime.launch"):
        build_launch({"runtime": {}}, "/x.gguf")


def test_sha256_file(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"frontier")
    assert sha256_file(f) == hashlib.sha256(b"frontier").hexdigest()


def _profile_with_pin(name: str, digest: str) -> dict:
    return {
        "artifacts": [
            {
                "quant": "q4_test",
                "sha256": None,
                "shards": [{"path": f"dir/{name}", "sha256": digest}],
            }
        ]
    }


def test_verify_artifact_accepts_matching_hash(tmp_path):
    f = tmp_path / "shard-00001.gguf"
    f.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    message = verify_artifact(f, _profile_with_pin(f.name, digest), "q4_test")
    assert "verified" in message


def test_verify_artifact_rejects_mismatch(tmp_path):
    f = tmp_path / "shard-00001.gguf"
    f.write_bytes(b"tampered")
    good = hashlib.sha256(b"weights").hexdigest()
    with pytest.raises(RunError, match="mismatch"):
        verify_artifact(f, _profile_with_pin(f.name, good), "q4_test")


def test_verify_artifact_requires_pins(tmp_path):
    f = tmp_path / "shard.gguf"
    f.write_bytes(b"x")
    profile = {"artifacts": [{"quant": "q4_test", "sha256": None, "shards": []}]}
    with pytest.raises(RunError, match="no sha256 pins"):
        verify_artifact(f, profile, "q4_test")


def test_verify_artifact_rejects_unknown_filename(tmp_path):
    f = tmp_path / "unrelated.gguf"
    f.write_bytes(b"x")
    digest = hashlib.sha256(b"x").hexdigest()
    with pytest.raises(RunError, match="does not match any pinned"):
        verify_artifact(f, _profile_with_pin("other.gguf", digest), "q4_test")
