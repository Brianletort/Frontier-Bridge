"""Planner v0: fit checks, refusal behavior, tiering, and schema-valid output."""

import pytest

from frontier_bridge.planner.engine import PlanError, generate_plan
from frontier_bridge.validation import validate_instance


def test_glm_on_m5_max_is_experimental_streaming_plan(repo_root):
    plan = generate_plan(
        repo_root, "glm-5.2", "m5_max_128gb", "coding_agent", 32768
    )
    assert plan["verdict"] == "experimental"
    assert plan["placement"]["resident"]["dense_core"] == "unified0"
    assert plan["placement"]["tiered"]["routed_experts"]["l2"]["mode"] == "stream_on_miss"
    assert plan["runtime"]["engine"] == "ds4"
    assert "decode_latency_spikes_on_expert_miss" in plan["risks"]
    assert "model_size_estimated_from_param_count_not_measured" in plan["risks"]
    # Expected values are never hand-typed: unmeasured means null.
    assert plan["expected"]["decode_tps"]["p50"] is None
    assert plan["expected"]["usability_class"] == "unrated"


def test_glm_on_rtx6000_uses_three_tiers_with_pinned_ram(repo_root):
    plan = generate_plan(
        repo_root, "glm-5.2", "rtx6000_96gb_64ram", "coding_agent", 131072
    )
    tiers = plan["placement"]["tiered"]["routed_experts"]
    assert tiers["l0"]["node"] == "vram0"
    assert tiers["l1"]["node"] == "sysram0"
    assert tiers["l1"]["pinned"] is True
    assert tiers["l2"]["node"] == "nvme0"
    kv = plan["placement"]["tiered"]["kv_cache"]
    assert kv["hot"]["window_tokens"] == 32768  # capped below the 128k budget
    assert kv["warm"]["node"] == "sysram0"


def test_unverified_model_is_refused_not_guessed(repo_root):
    plan = generate_plan(
        repo_root, "deepseek-v4-flash", "m5_max_128gb", "chat", 32768
    )
    assert plan["verdict"] == "not_recommended"
    assert any("insufficient_model_data" in r for r in plan["reasons"])
    assert "placement" not in plan


def test_context_over_claimed_max_is_refused(repo_root):
    plan = generate_plan(
        repo_root, "glm-5.2", "m5_max_128gb", "long_context", 2_000_000
    )
    assert plan["verdict"] == "not_recommended"
    assert any("context_budget_exceeds_claimed_max" in r for r in plan["reasons"])


def test_forced_quant_is_respected(repo_root):
    plan = generate_plan(
        repo_root, "glm-5.2", "m5_max_128gb", "chat", 8192, quant="q2_routed"
    )
    assert plan["inputs"]["modelprofile"] == "glm-5.2/q2_routed"


def test_unknown_inputs_raise_plan_error(repo_root):
    with pytest.raises(PlanError, match="workload"):
        generate_plan(repo_root, "glm-5.2", "m5_max_128gb", "gaming", 8192)
    with pytest.raises(PlanError, match="hardware"):
        generate_plan(repo_root, "glm-5.2", "no_such_box", "chat", 8192)
    with pytest.raises(PlanError, match="model"):
        generate_plan(repo_root, "no-such-model", "m5_max_128gb", "chat", 8192)
    with pytest.raises(PlanError, match="quant"):
        generate_plan(repo_root, "glm-5.2", "m5_max_128gb", "chat", 8192, quant="q9")


@pytest.mark.parametrize(
    ("model", "hardware", "workload", "ctx"),
    [
        ("glm-5.2", "m5_max_128gb", "coding_agent", 32768),
        ("glm-5.2", "rtx6000_96gb_64ram", "coding_agent", 131072),
        ("glm-5.2", "gb10_128gb", "chat", 32768),
        ("deepseek-v4-flash", "m5_max_128gb", "chat", 32768),
        ("glm-5.2", "m5_max_128gb", "long_context", 2_000_000),
    ],
)
def test_every_planner_output_validates_against_plan_v1(
    repo_root, model, hardware, workload, ctx
):
    plan = generate_plan(repo_root, model, hardware, workload, ctx)
    assert validate_instance(plan) == []
