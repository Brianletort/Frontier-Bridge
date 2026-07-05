"""Web UI API tested against a live local server instance."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from frontier_bridge.ui import SetupJob, make_handler, make_server


@pytest.fixture()
def ui_server(repo_root):
    server = make_server(repo_root=repo_root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _post(url: str, payload: dict):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def test_index_serves_html(ui_server):
    with urllib.request.urlopen(ui_server + "/", timeout=30) as r:
        assert r.status == 200
        body = r.read().decode("utf-8")
    assert "Frontier Bridge" in body


def test_state_lists_catalog(ui_server):
    status, state = _get(ui_server + "/api/state")
    assert status == 200
    hardware_ids = {h["profile_id"] for h in state["hardware"]}
    assert "apple_m5_max_137gb_detected" in hardware_ids
    model_ids = {m["model_id"] for m in state["models"]}
    assert {"glm-5.2", "deepseek-v4-flash", "kimi-k2.6"} <= model_ids
    # Verified rows are surfaced for the matrix panel.
    assert any(r["status"] == "verified" for r in state["matrix"])


def test_plan_endpoint_returns_plan(ui_server):
    status, plan = _get(
        ui_server
        + "/api/plan?model=deepseek-v4-flash&hardware=apple_m5_max_137gb_detected"
        + "&workload=chat&ctx=8192&quant=q2_imatrix"
    )
    assert status == 200
    assert plan["schema_version"] == "plan/v1"
    assert plan["verdict"] in ("recommended", "experimental")


def test_plan_endpoint_reports_errors(ui_server):
    try:
        _get(ui_server + "/api/plan?model=nope&hardware=nope&workload=chat&ctx=1")
        assert False, "expected HTTP 400"
    except urllib.error.HTTPError as exc:
        assert exc.code == 400
        assert "error" in json.loads(exc.read().decode("utf-8"))


def test_setup_rejects_refusal_plans(ui_server):
    refusal = {"schema_version": "plan/v1", "verdict": "not_recommended", "inputs": {}}
    try:
        _post(ui_server + "/api/setup", {"plan": refusal})
        assert False, "expected HTTP 400"
    except urllib.error.HTTPError as exc:
        assert exc.code == 400


def test_job_endpoint_idle_by_default(ui_server):
    status, job = _get(ui_server + "/api/job")
    assert status == 200
    assert job["status"] == "idle"


def test_setup_job_log_is_bounded():
    job = SetupJob()
    for i in range(1000):
        job.echo(f"line {i}")
    snap = job.snapshot()
    assert len(snap["log"]) == 100
    assert snap["log"][-1] == "line 999"


def test_make_handler_reads_bundled_frontend(repo_root):
    handler = make_handler(repo_root, SetupJob())
    assert handler is not None
