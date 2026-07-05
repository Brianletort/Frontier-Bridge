"""Planner v0: fit checks, refusal behavior, tiering, and schema-valid output."""

import pytest

from frontier_bridge.planner.engine import PlanError, generate_plan
from frontier_bridge.validation import validate_instance


def test_glm_on_m5_max_is_experimental_streaming_plan(repo_root):
    plan = generate_plan(
        repo_root, "glm-5.2", "m5_max_128gb", "coding_agent", 32768
    )
    # GLM Q4 is 467GB measured vs 128GB unified: streaming, never "recommended".
    assert plan["verdict"] == "experimental"
    assert plan["placement"]["resident"]["dense_core"] == "unified0"
    assert plan["placement"]["tiered"]["routed_experts"]["l2"]["mode"] == "stream_on_miss"
    assert plan["runtime"]["engine"] == "ds4"
    assert "decode_latency_spikes_on_expert_miss" in plan["risks"]
    # Size is now measured (pinned artifact), so the estimate risk must be gone.
    assert "model_size_estimated_from_param_count_not_measured" not in plan["risks"]
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


def test_unverified_model_is_refused_not_guessed(repo_root, tmp_path):
    """A model profile with null params must be refused, not sized hopefully."""
    import shutil

    stub_root = tmp_path
    shutil.copytree(repo_root / "hardware_profiles", stub_root / "hardware_profiles")
    model_dir = stub_root / "model_profiles" / "stub"
    model_dir.mkdir(parents=True)
    (model_dir / "q4.yaml").write_text(
        """schema_version: modelprofile/v1
model_id: stub-model
family: stub
architecture:
  type: moe
  params_total_b: null
artifacts:
  - { format: gguf, quant: q4_stub, size_gb: null, source: null, sha256: null }
runtime_support:
  - { runtime: llama_cpp, status: claimed }
memory_model: {}
""",
        encoding="utf-8",
    )
    plan = generate_plan(stub_root, "stub-model", "m5_max_128gb", "chat", 32768)
    assert plan["verdict"] == "not_recommended"
    assert any("insufficient_model_data" in r for r in plan["reasons"])
    assert "placement" not in plan


def test_llama_cpp_offload_computed_from_measured_sizes(repo_root):
    """GLM Q4: 446GB routed vs ~115GB usable unified -> most MoE layers offload."""
    plan = generate_plan(
        repo_root, "glm-5.2", "apple_m5_max_137gb_detected", "coding_agent", 32768,
        quant="q4_routed", engine_override="llama_cpp",
    )
    launch = plan["runtime"]["launch"]
    assert "--n-cpu-moe" in launch
    n = int(launch.split("--n-cpu-moe")[1].split()[0])
    assert 50 <= n <= 79  # bounded by the 79 measured MoE layers

    # A model that fits resident gets no offload flag.
    fits = generate_plan(
        repo_root, "deepseek-v4-flash", "apple_m5_max_137gb_detected", "chat", 16384,
        quant="q2_imatrix", engine_override="llama_cpp",
    )
    assert "--n-cpu-moe" not in fits["runtime"]["launch"]


def test_deepseek_q2_fits_m5_max_and_is_recommended(repo_root):
    """107GB measured Q2 in 128GB unified memory: a real, measured 'yes'."""
    plan = generate_plan(
        repo_root, "deepseek-v4-flash", "m5_max_128gb", "chat", 32768, quant="q2_imatrix"
    )
    assert plan["verdict"] == "recommended"
    assert "decode_latency_spikes_on_expert_miss" not in plan["risks"]


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
