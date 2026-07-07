"""Runbook matching, provenance verification, and rendering (RFC 0003)."""

import copy

import yaml

from frontier_bridge.runbook import (
    load_runbooks,
    match_runbook,
    render_markdown,
    verify_runbook,
)


def _load_committed(repo_root):
    entries = load_runbooks(repo_root)
    assert entries, "at least one runbook must be committed"
    return entries


def _unified_runbook(repo_root):
    return next(
        e.data for e in _load_committed(repo_root) if e.runbook_id == "unified-128gb"
    )


def test_committed_runbooks_pass_verify(repo_root):
    for entry in _load_committed(repo_root):
        assert verify_runbook(entry.data, repo_root) == [], entry.path


def test_m5_max_matches_unified_class(repo_root):
    profile = yaml.safe_load(
        (repo_root / "hardware_profiles" / "apple_m5_max_137gb_detected.yaml").read_text()
    )
    report = match_runbook(_unified_runbook(repo_root), profile)
    assert report.matched, report.unmet


def test_discrete_gpu_box_does_not_match_unified_class(repo_root):
    profile = yaml.safe_load(
        (repo_root / "hardware_profiles" / "rtx6000_96gb_64ram.yaml").read_text()
    )
    report = match_runbook(_unified_runbook(repo_root), profile)
    assert not report.matched
    assert report.unmet


def test_unmeasured_field_fails_minimum_bound_predicate(repo_root):
    """Honest degradation: a machine with an unmeasured field matches fewer
    runbooks, never more."""
    runbook = copy.deepcopy(_unified_runbook(repo_root))
    runbook["hardware_class"]["require"].append(
        {"kind": "storage", "measured": {"seq_read_gbps": {"min": 1.0}}}
    )
    profile = {
        "nodes": [
            {"id": "unified0", "kind": "memory", "class": "unified", "capacity_gb": 128},
            {"id": "gpu0", "kind": "compute", "class": "gpu"},
            {"id": "ssd0", "kind": "storage", "class": "nvme", "measured": {"seq_read_gbps": None}},
        ]
    }
    report = match_runbook(runbook, profile)
    assert not report.matched


def test_verify_catches_number_drift(repo_root):
    """A runbook number that disagrees with its source benchresult must fail."""
    runbook = copy.deepcopy(_unified_runbook(repo_root))
    measured = [e for e in runbook["expected"] if not e.get("unmeasured")]
    assert measured, "fixture runbook needs at least one measured entry"
    measured[0]["decode_tps"] = 99.9
    errors = verify_runbook(runbook, repo_root)
    assert any("disagrees with benchresult" in e for e in errors)


def test_verify_catches_missing_source_result(repo_root):
    runbook = copy.deepcopy(_unified_runbook(repo_root))
    measured = [e for e in runbook["expected"] if not e.get("unmeasured")]
    measured[0]["source"] = "no-such-result"
    errors = verify_runbook(runbook, repo_root)
    assert any("not a committed benchresult" in e for e in errors)


def test_verify_catches_verdict_disagreement_with_plan(repo_root):
    runbook = copy.deepcopy(_unified_runbook(repo_root))
    runbook["model_menu"][0]["verdict"] = "not_recommended"
    errors = verify_runbook(runbook, repo_root)
    assert any("disagrees with plan" in e for e in errors)


def test_render_carries_sources_not_just_numbers(repo_root):
    runbook = _unified_runbook(repo_root)
    markdown = render_markdown(runbook)
    for entry in runbook["expected"]:
        if not entry.get("unmeasured"):
            assert entry["source"] in markdown
    assert "unmeasured" in markdown
    assert "do not edit by hand" in markdown
