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


def _minimal_runbook() -> dict:
    return {
        "schema_version": "runbook/v1",
        "runbook_id": "unified-128gb",
        "title": "Unified 128 GB class",
        "status": "draft",
        "hardware_class": {
            "name": "unified_128gb",
            "require": [
                {"kind": "memory", "class": ["unified"], "capacity_gb": {"min": 110, "max": 145}}
            ],
        },
        "diagnosis": [{"step": "Run detect", "command": "frontier detect"}],
        "model_menu": [
            {
                "modelprofile": "deepseek-v4-flash/q2_imatrix",
                "verdict": "recommended",
                "plan_ref": "plans/m5max_dsv4flash_q2_chat_16k.yaml",
                "role": "daily_driver",
            }
        ],
        "provenance": {"authored_by": "test", "created": "2026-07-06"},
    }


def test_runbook_minimal_validates():
    assert validate_instance(_minimal_runbook()) == []


def test_runbook_expected_numbers_require_source():
    """A runbook performance entry without a benchresult source is invalid (RFC 0003)."""
    runbook = _minimal_runbook()
    runbook["expected"] = [
        {
            "modelprofile": "deepseek-v4-flash/q2_imatrix",
            "workload": "chat",
            "decode_tps": 7.44,
        }
    ]
    errors = validate_instance(runbook)
    assert any("source" in e for e in errors)
    runbook["expected"][0]["source"] = "m5max-dsv4q2-chat-run3"
    assert validate_instance(runbook) == []


def test_runbook_unmeasured_entry_carries_no_numbers():
    runbook = _minimal_runbook()
    runbook["expected"] = [
        {
            "modelprofile": "glm-5.2/q2_routed",
            "workload": "coding_agent",
            "unmeasured": True,
            "decode_tps": 5.0,
        }
    ]
    errors = validate_instance(runbook)
    assert errors, "unmeasured entries must not carry performance numbers"
    runbook["expected"][0]["decode_tps"] = None
    assert validate_instance(runbook) == []


def test_modelprofile_catalog_block_validates():
    """RFC 0004 additive catalog block: valid statuses only, size_class required."""
    profile = {
        "schema_version": "modelprofile/v1",
        "model_id": "test-model",
        "family": "test",
        "architecture": {"type": "moe", "params_total_b": 300},
        "artifacts": [{"format": "gguf", "quant": "q2"}],
        "runtime_support": [{"runtime": "llama_cpp", "status": "claimed"}],
        "memory_model": {},
        "catalog": {"status": "admitted", "size_class": "mid"},
    }
    assert validate_instance(profile) == []
    profile["catalog"]["status"] = "featured"
    assert validate_instance(profile)
    profile["catalog"] = {"status": "admitted"}
    assert any("size_class" in e for e in validate_instance(profile))


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
