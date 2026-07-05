"""`frontier setup` provisioning: shard resolution, space checks, and resumable
hash-verified downloads against a local Range-capable HTTP server."""

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from frontier_bridge.provision import (
    SetupError,
    ShardSpec,
    check_free_space,
    download_shard,
    install_hint,
    resolve_shards,
    runtime_binary,
)

_PAYLOAD = b"frontier-bridge test payload " * 4096  # ~116KB
_PAYLOAD_SHA = hashlib.sha256(_PAYLOAD).hexdigest()


class _RangeHandler(BaseHTTPRequestHandler):
    """Static payload server honoring HTTP Range requests."""

    def do_GET(self):
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            start = int(range_header[len("bytes="):].split("-", 1)[0])
            body = _PAYLOAD[start:]
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{len(_PAYLOAD) - 1}/{len(_PAYLOAD)}"
            )
        else:
            body = _PAYLOAD
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def payload_server():
    server = HTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/model.gguf"
    server.shutdown()


_PROFILE = {
    "artifacts": [
        {
            "quant": "q2_test",
            "size_gb": 0.0001,
            "source": "https://huggingface.co/org/repo/tree/main/Q2",
            "sha256": _PAYLOAD_SHA,
            "shards": [
                {"path": "Q2/model-00001-of-00002.gguf", "sha256": _PAYLOAD_SHA},
                {"path": "Q2/model-00002-of-00002.gguf", "sha256": _PAYLOAD_SHA},
            ],
        },
        {"quant": "q_unpinned", "size_gb": 1.0, "source": None, "sha256": None},
    ]
}


def test_resolve_shards_builds_resolve_urls():
    specs = resolve_shards(_PROFILE, "q2_test")
    assert len(specs) == 2
    assert specs[0].url == (
        "https://huggingface.co/org/repo/resolve/main/Q2/model-00001-of-00002.gguf"
    )
    assert specs[0].filename == "model-00001-of-00002.gguf"
    assert specs[0].sha256 == _PAYLOAD_SHA


def test_resolve_shards_refuses_sourceless_artifact():
    with pytest.raises(SetupError, match="no downloadable source"):
        resolve_shards(_PROFILE, "q_unpinned")
    with pytest.raises(SetupError, match="no artifact"):
        resolve_shards(_PROFILE, "missing_quant")


def test_check_free_space(tmp_path: Path):
    check_free_space(tmp_path, None)  # unknown size: no refusal here, risk disclosed elsewhere
    check_free_space(tmp_path, 0.001)
    with pytest.raises(SetupError, match="not enough free space"):
        check_free_space(tmp_path, 10_000_000)  # 10 exabytes


def test_download_verifies_and_renames(tmp_path: Path, payload_server):
    spec = ShardSpec(url=payload_server, filename="model.gguf", sha256=_PAYLOAD_SHA)
    path = download_shard(spec, tmp_path, echo=lambda s: None)
    assert path == tmp_path / "model.gguf"
    assert path.read_bytes() == _PAYLOAD
    assert not (tmp_path / "model.gguf.part").exists()


def test_download_resumes_from_part_file(tmp_path: Path, payload_server):
    (tmp_path / "model.gguf.part").write_bytes(_PAYLOAD[: len(_PAYLOAD) // 2])
    spec = ShardSpec(url=payload_server, filename="model.gguf", sha256=_PAYLOAD_SHA)
    path = download_shard(spec, tmp_path, echo=lambda s: None)
    # The hash only passes if the resumed second half lined up exactly.
    assert path.read_bytes() == _PAYLOAD


def test_download_refuses_unpinned_by_default(tmp_path: Path, payload_server):
    spec = ShardSpec(url=payload_server, filename="model.gguf", sha256=None)
    with pytest.raises(SetupError, match="no sha256 pin"):
        download_shard(spec, tmp_path, echo=lambda s: None)
    # Explicit override downloads without verification.
    path = download_shard(spec, tmp_path, echo=lambda s: None, allow_unpinned=True)
    assert path.read_bytes() == _PAYLOAD


def test_download_rejects_corrupt_payload(tmp_path: Path, payload_server):
    spec = ShardSpec(url=payload_server, filename="model.gguf", sha256="0" * 64)
    with pytest.raises(SetupError, match="sha256 mismatch"):
        download_shard(spec, tmp_path, echo=lambda s: None)
    # The .part stays for inspection; no verified-looking final file appears.
    assert (tmp_path / "model.gguf.part").exists()
    assert not (tmp_path / "model.gguf").exists()


def test_existing_verified_file_is_not_redownloaded(tmp_path: Path):
    (tmp_path / "model.gguf").write_bytes(_PAYLOAD)
    spec = ShardSpec(url="http://127.0.0.1:1/unreachable", filename="model.gguf", sha256=_PAYLOAD_SHA)
    path = download_shard(spec, tmp_path, echo=lambda s: None)
    assert path.read_bytes() == _PAYLOAD


def test_runtime_binary_map():
    assert runtime_binary("llama_cpp") == "llama-server"
    assert runtime_binary("nope") is None
    assert "brew install llama.cpp" in install_hint("llama_cpp")
