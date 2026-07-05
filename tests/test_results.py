"""Results folding and matrix rendering."""

import json

from frontier_bridge.bench.engine import suggest_usability
from frontier_bridge.results import fold_matrix, load_results, render_markdown


def _result(result_id, status="claimed", repros=0, decode=12.5, hw="rtx6000", model="glm-5.2/q4"):
    return {
        "schema_version": "benchresult/v1",
        "result_id": result_id,
        "measured_at": "2026-07-04T00:00:00Z",
        "pins": {
            "plan_hash": "p", "model_sha256": "m",
            "runtime_commit": "r", "hwprofile_id": hw,
        },
        "metrics": {
            "decode_tps": decode,
            "ttft_ms": 900.0,
            "token_latency_ms": {"p50": 70.0, "p95": 120.0, "p99": 300.0},
            "context_len_tokens": 32768,
        },
        "workload_tasks": [{"task": "t", "passed": True, "notes": None}],
        "reproductions": ["a", "b"][:repros],
        "status": status,
        "subject": {"modelprofile": model, "workload": "coding_agent"},
        "usability_suggested": "usable",
    }


def test_fold_prefers_verified_and_reproduced():
    weak = _result("weak", status="claimed", repros=0)
    strong = _result("strong", status="verified", repros=2)
    rows = fold_matrix([weak, strong])
    assert len(rows) == 1
    assert rows[0].result_id == "strong"
    assert rows[0].status == "verified"


def test_fold_groups_by_hw_model_workload():
    rows = fold_matrix([
        _result("a", hw="rtx6000"),
        _result("b", hw="gb10_128gb"),
        _result("c", hw="rtx6000", model="deepseek-v4-flash/q2"),
    ])
    assert len(rows) == 3


def test_render_markdown_includes_all_rows_and_caveat():
    markdown = render_markdown(fold_matrix([_result("x")]))
    assert "| rtx6000 |" in markdown
    assert "glm-5.2/q4" in markdown
    assert "verified" in markdown  # the caveat line
    assert render_markdown([]).startswith("No benchmark results")


def test_load_results_scans_dirs(tmp_path):
    d = tmp_path / "results" / "community"
    d.mkdir(parents=True)
    (d / "r1.json").write_text(json.dumps(_result("r1")), encoding="utf-8")
    (d / "junk.json").write_text("{not json", encoding="utf-8")
    (d / "other.json").write_text(json.dumps({"schema_version": "other/v1"}), encoding="utf-8")
    results = load_results(tmp_path)
    assert [r["result_id"] for r in results] == ["r1"]


def test_suggest_usability_thresholds():
    good = _result("g")
    assert suggest_usability(good, "coding_agent") == "agent_capable"
    assert suggest_usability(good, "chat") == "interactive"

    slow = _result("s")
    slow["metrics"]["token_latency_ms"]["p95"] = 900.0
    assert suggest_usability(slow, "chat") == "usable"

    failing = _result("f")
    failing["workload_tasks"] = [
        {"task": "t1", "passed": False, "notes": None},
        {"task": "t2", "passed": True, "notes": None},
    ]
    assert suggest_usability(failing, "chat") == "runs"

    broken = _result("b")
    broken["workload_tasks"] = [{"task": "t", "passed": False, "notes": "URLError: dead"}]
    assert suggest_usability(broken, "chat") == "not_recommended"
