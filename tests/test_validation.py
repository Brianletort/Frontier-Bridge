"""Schema validation round-trips for all committed YAML plus negative cases."""

from frontier_bridge.validation import validate_instance, validate_path


def test_all_committed_yaml_validates(repo_root):
    report = validate_path(repo_root)
    assert report.ok, [f"{i.path}: {i.message}" for i in report.issues]
    # The reference profiles and templates must actually be checked, not skipped.
    assert len(report.checked) >= 8


def test_missing_schema_version_is_reported():
    assert validate_instance({"profile_id": "x"}) == [
        "missing or non-string 'schema_version' field"
    ]


def test_unknown_schema_version_rejected_not_guessed():
    errors = validate_instance({"schema_version": "hwprofile/v99"})
    assert len(errors) == 1
    assert "Unknown schema_version" in errors[0]


def test_hwprofile_requires_nodes():
    errors = validate_instance(
        {
            "schema_version": "hwprofile/v1",
            "profile_id": "test_box",
            "provenance": {"method": "manual"},
            "nodes": [],
            "links": [],
        }
    )
    assert any("nodes" in e for e in errors)


def test_hwprofile_rejects_invalid_provenance_method():
    errors = validate_instance(
        {
            "schema_version": "hwprofile/v1",
            "profile_id": "test_box",
            "provenance": {"method": "guessed"},
            "nodes": [{"id": "cpu0", "kind": "compute", "class": "cpu"}],
            "links": [],
        }
    )
    assert any("guessed" in e for e in errors)


def test_refusal_plan_requires_reasons():
    plan = {
        "schema_version": "plan/v1",
        "plan_id": "test-refusal",
        "inputs": {
            "hwprofile": "x",
            "modelprofile": "y/q4",
            "workload": "chat",
            "context_budget": 1024,
        },
        "verdict": "not_recommended",
    }
    errors = validate_instance(plan)
    assert any("reasons" in e for e in errors)
    plan["reasons"] = ["model_does_not_fit"]
    assert validate_instance(plan) == []


def test_plan_rejects_non_open_cache_policy():
    """Only IP-cleared policies are valid plan values in v1 (see IP_NOTICE.md)."""
    plan = {
        "schema_version": "plan/v1",
        "plan_id": "test-policy",
        "inputs": {
            "hwprofile": "x",
            "modelprofile": "y/q4",
            "workload": "chat",
            "context_budget": 1024,
        },
        "verdict": "experimental",
        "placement": {
            "resident": {"dense_core": "vram0"},
            "tiered": {
                "routed_experts": {
                    "l0": {"node": "vram0", "policy": "router_aware_prefetch"}
                }
            },
        },
        "phases": {},
        "runtime": {"engine": "ds4"},
        "expected": {"usability_class": "unrated"},
    }
    errors = validate_instance(plan)
    assert any("router_aware_prefetch" in e for e in errors)


def test_benchresult_requires_all_pins():
    errors = validate_instance(
        {
            "schema_version": "benchresult/v1",
            "result_id": "run-001",
            "measured_at": "2026-07-04T00:00:00Z",
            "pins": {"plan_hash": None},
            "metrics": {},
        }
    )
    assert any("model_sha256" in e for e in errors)
