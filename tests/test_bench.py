"""Bench harness: suite loading, task checks, result assembly, verified-status
rule, and an end-to-end run against a mock OpenAI-compatible SSE server."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from frontier_bridge.bench.client import CompletionTiming, percentile, stream_completion
from frontier_bridge.bench.engine import (
    SUITES,
    build_result,
    check_task,
    finalize_status,
    load_suite,
    plan_hash,
    run_suite,
)
from frontier_bridge.validation import validate_instance


def test_bundled_suites_load_and_expand():
    for name in SUITES:
        prompts = load_suite(name)
        assert prompts, name
        for prompt in prompts:
            assert prompt["messages"], f"{name}: unexpanded prompt {prompt.get('id')}"
    # Long-context expansion produces genuinely long input with the needle inside.
    long_prompts = load_suite("long_context")
    needle_prompt = next(p for p in long_prompts if p["id"] == "long-needle-16k")
    content = needle_prompt["messages"][0]["content"]
    assert "BLUEHERON-42" in content
    assert len(content) > 50_000


def test_percentile():
    assert percentile([], 95) is None
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == 51.0
    assert percentile(values, 99) == 100.0


def test_check_task_variants():
    ok = CompletionTiming(prompt_id="x", output_text="The answer is BLUEHERON-42.")
    ok.completion_tokens = 12
    assert check_task({"check": {"type": "contains_any", "values": ["blueheron-42"]}}, ok)
    assert not check_task({"check": {"type": "contains_all", "values": ["missing"]}}, ok)
    assert check_task({"check": {"type": "min_tokens", "value": 10}}, ok)
    assert check_task({}, ok)
    failed = CompletionTiming(prompt_id="y", error="URLError: boom")
    assert not check_task({}, failed)


def test_verified_status_requires_pins_and_two_reproductions():
    base = {
        "pins": {
            "plan_hash": "a", "model_sha256": "b",
            "runtime_commit": "c", "hwprofile_id": "d",
        },
        "reproductions": [],
    }
    assert finalize_status(dict(base))["status"] == "claimed"  # no reproductions
    one = {**base, "reproductions": ["r1"]}
    assert finalize_status(one)["status"] == "claimed"
    two = {**base, "reproductions": ["r1", "r2"]}
    assert finalize_status(two)["status"] == "verified"
    unpinned = {**two, "pins": {**base["pins"], "runtime_commit": None}}
    assert finalize_status(unpinned)["status"] == "claimed"


def test_plan_hash_is_stable_and_order_independent():
    assert plan_hash({"a": 1, "b": 2}) == plan_hash({"b": 2, "a": 1})
    assert plan_hash({"a": 1}) != plan_hash({"a": 2})


class _MockSSEHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible streaming endpoint."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        words = ["The", " answer", " is", " 42", "."]
        for word in words:
            event = {"choices": [{"delta": {"content": word}}]}
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.02)  # force distinct packets so decode timing is measurable
        usage = {"choices": [], "usage": {"completion_tokens": 5, "prompt_tokens": 20}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"data": [{"id": "mock-model"}]}')

    def log_message(self, *args):
        pass


@pytest.fixture()
def mock_server():
    server = HTTPServer(("127.0.0.1", 0), _MockSSEHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_port
    server.shutdown()


def test_stream_completion_measures_timing(mock_server):
    timing = stream_completion(
        mock_server, [{"role": "user", "content": "What is 6x7?"}], prompt_id="t1"
    )
    assert timing.error is None
    assert timing.ttft_ms is not None and timing.ttft_ms >= 0
    assert timing.completion_tokens == 5
    assert timing.prompt_tokens == 20
    assert "42" in timing.output_text
    assert timing.decode_tps is not None


def test_end_to_end_benchresult_validates(mock_server):
    prompts = [
        {
            "id": "e2e-1",
            "messages": [{"role": "user", "content": "answer?"}],
            "check": {"type": "contains_any", "values": ["42"]},
        }
    ]
    timings, tasks = run_suite(mock_server, "custom", prompts)
    assert tasks[0]["passed"] is True
    result = build_result(
        result_id="e2e-test-001",
        suite_name="custom",
        timings=timings,
        tasks=tasks,
        telemetry={"peak_vram_gb": None, "peak_ram_gb": 1.2, "ssd_total_read_gb": 0.0},
        pins={
            "plan_hash": "x", "model_sha256": None,
            "runtime_commit": None, "hwprofile_id": "m5_max_128gb",
        },
        context_len=8192,
    )
    assert validate_instance(result) == []
    assert result["status"] == "claimed"  # unpinned + no reproductions
    assert result["metrics"]["decode_tps"] is not None
