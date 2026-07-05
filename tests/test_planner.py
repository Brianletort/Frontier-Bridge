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

    # DeepSeek Q2 (107GB) on 137GB unified sits just over the Metal working-set
    # headroom: a small offload is computed, far below the GLM case.
    near_fit = generate_plan(
        repo_root, "deepseek-v4-flash", "apple_m5_max_137gb_detected", "chat", 16384,
        quant="q2_imatrix", engine_override="llama_cpp",
    )
    launch2 = near_fit["runtime"]["launch"]
    if "--n-cpu-moe" in launch2:
        n2 = int(launch2.split("--n-cpu-moe")[1].split()[0])
        assert 1 <= n2 <= 10


def test_tiering_flags_enforced_in_launch_command(repo_root):
    """Tier placement is applied to the launch command, not left advisory."""
    # DeepSeek Q2 (107GB) on RTX 6000 (96GB VRAM + 64GB pinnable RAM): fits in
    # combined memory, experts overflow VRAM -> offload + mlock the L1 overflow.
    pinned = generate_plan(
        repo_root, "deepseek-v4-flash", "rtx6000_96gb_64ram", "chat", 16384,
        quant="q2_imatrix", engine_override="llama_cpp",
    )
    flags = pinned["runtime"]["tiering_flags"]
    assert flags["mlock"] is True
    assert flags["mmap_stream_l2"] is False
    assert "--mlock" in pinned["runtime"]["launch"]
    assert "--n-cpu-moe" in pinned["runtime"]["launch"]

    # GLM Q4 (467GB) on the same box streams from NVMe: mmap is the L2 tier and
    # mlock must never be applied.
    streaming = generate_plan(
        repo_root, "glm-5.2", "rtx6000_96gb_64ram", "chat", 16384,
        quant="q4_routed", engine_override="llama_cpp",
    )
    flags = streaming["runtime"]["tiering_flags"]
    assert flags["mmap_stream_l2"] is True
    assert flags["mlock"] is False
    assert "--mlock" not in streaming["runtime"]["launch"]

    # M5 Max unified memory has no separate pinnable system pool: no mlock.
    unified = generate_plan(
        repo_root, "deepseek-v4-flash", "apple_m5_max_137gb_detected", "chat", 16384,
        quant="q2_imatrix", engine_override="llama_cpp",
    )
    assert unified["runtime"]["tiering_flags"]["mlock"] is False


def test_expected_filled_from_verified_results_never_hand_typed(repo_root):
    """The M5 Max DeepSeek Q2 chat combination has committed verified results:
    the plan's expected block must cite them; unmeasured combos stay null."""
    plan = generate_plan(
        repo_root, "deepseek-v4-flash", "apple_m5_max_137gb_detected", "chat", 8192,
        quant="q2_imatrix",
    )
    expected = plan["expected"]
    assert expected["decode_tps"]["p50"] is not None
    assert expected["decode_tps"]["source"].startswith("m5max-dsv4q2-chat")
    assert expected["usability_class"] == "interactive"

    unmeasured = generate_plan(
        repo_root, "deepseek-v4-flash", "gb10_128gb", "chat", 8192, quant="q2_imatrix"
    )
    assert unmeasured["expected"]["decode_tps"]["p50"] is None
    assert unmeasured["expected"]["usability_class"] == "unrated"


def test_measured_kv_budget_shapes_plan(repo_root, tmp_path):
    """When kv_per_1k_tokens_mb is measured, the hot KV footprint is sized,
    it comes out of the L0 expert budget, and the unmeasured-KV risk is gone."""
    import shutil

    import yaml

    stub_root = tmp_path
    shutil.copytree(repo_root / "hardware_profiles", stub_root / "hardware_profiles")
    src = repo_root / "model_profiles" / "deepseek_v4_flash" / "q2_imatrix.yaml"
    data = yaml.safe_load(src.read_text())
    data["memory_model"]["kv_per_1k_tokens_mb"] = 100.0
    model_dir = stub_root / "model_profiles" / "deepseek_v4_flash"
    model_dir.mkdir(parents=True)
    (model_dir / "q2_imatrix.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    plan = generate_plan(
        stub_root, "deepseek-v4-flash", "apple_m5_max_137gb_detected", "chat", 8192,
        quant="q2_imatrix",
    )
    hot = plan["placement"]["tiered"]["kv_cache"]["hot"]
    assert hot["estimated_gb"] == round(100.0 * 8192 / 1000 / 1024, 2)
    assert hot["source"] == "measured_kv_per_1k_tokens"
    assert "kv_footprint_unmeasured_context_budget_not_validated" not in plan["risks"]

    # Same plan without the measurement: bigger L0 budget, disclosed risk.
    baseline = generate_plan(
        repo_root, "deepseek-v4-flash", "apple_m5_max_137gb_detected", "chat", 8192,
        quant="q2_imatrix",
    )
    assert "kv_footprint_unmeasured_context_budget_not_validated" in baseline["risks"]
    l0_with_kv = plan["placement"]["tiered"]["routed_experts"]["l0"]["budget_gb"]
    l0_without = baseline["placement"]["tiered"]["routed_experts"]["l0"]["budget_gb"]
    assert l0_with_kv < l0_without


def test_ingested_profiles_plan_end_to_end(repo_root):
    """Profiles generated by `frontier catalog add` (measured headers, LFS pins)
    must flow through the planner: small-enough models get recommended, 1T-class
    models get tiered/experimental — never a crash, never a guess."""
    small = generate_plan(
        repo_root, "gpt-oss-120b", "apple_m5_max_137gb_detected", "coding_agent", 32768
    )
    assert small["verdict"] == "recommended"

    big = generate_plan(
        repo_root, "kimi-k2.6", "apple_m5_max_137gb_detected", "chat", 8192
    )
    assert big["verdict"] == "experimental"
    assert big["inputs"]["modelprofile"].startswith("kimi-k2.6/")


def test_deepseek_q2_fits_m5_max_and_is_recommended(repo_root):
    """107GB measured Q2 in 128GB unified memory: a real, measured 'yes'."""
    plan = generate_plan(
        repo_root, "deepseek-v4-flash", "m5_max_128gb", "chat", 32768, quant="q2_imatrix"
    )
    assert plan["verdict"] == "recommended"
    assert "decode_latency_spikes_on_expert_miss" not in plan["risks"]


def test_external_ssd_ranks_below_internal_with_disclosed_prior(repo_root):
    """RFC 0002: with both drives unmeasured, the internal SSD outranks the
    Thunderbolt drive by documented class prior — and the plan says so."""
    plan = generate_plan(
        repo_root, "glm-5.2", "m5_max_128gb_tb5_ssd", "coding_agent", 32768
    )
    tiers = plan["placement"]["tiered"]["routed_experts"]
    assert tiers["l2"]["node"] == "ssd0"
    assert tiers["l3"]["node"] == "ssd1"
    assert tiers["l3"]["mode"] == "stream_on_miss"
    assert "tier_order_uses_class_priors_where_links_unmeasured" in plan["risks"]
    assert validate_instance(plan) == []


def test_measured_external_link_reorders_storage_tiers(repo_root, tmp_path):
    """RFC 0002: tiers are bandwidth classes, not device categories. A measured
    fast external link outranks an unmeasured internal drive — measurement
    decides, and no prior-order risk is recorded for the measured winner."""
    import shutil

    import yaml

    stub_root = tmp_path
    shutil.copytree(repo_root / "model_profiles", stub_root / "model_profiles")
    shutil.copytree(repo_root / "results", stub_root / "results")
    hw_dir = stub_root / "hardware_profiles"
    hw_dir.mkdir()
    src = repo_root / "hardware_profiles" / "m5_max_128gb_tb5_ssd.yaml"
    data = yaml.safe_load(src.read_text())
    for link in data["links"]:
        if link["from"] == "ssd1":
            link["measured"]["seq_read_gbps"] = 6.0  # measured TB5 drive
    (hw_dir / "m5_max_128gb_tb5_ssd.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False)
    )

    plan = generate_plan(
        stub_root, "glm-5.2", "m5_max_128gb_tb5_ssd", "coding_agent", 32768
    )
    tiers = plan["placement"]["tiered"]["routed_experts"]
    # 6.0 GB/s measured beats the internal drive's unmeasured 5.0 prior.
    assert tiers["l2"]["node"] == "ssd1"
    assert tiers["l3"]["node"] == "ssd0"


def test_egpu_becomes_resident_island_never_primary(repo_root):
    """RFC 0002: eGPU VRAM behind Thunderbolt is a resident island — a fixed
    expert subset lives there; it is never the primary pool and never a
    per-token streaming tier."""
    plan = generate_plan(
        repo_root, "glm-5.2", "rtx5090_128ram_egpu_4090", "coding_agent", 32768
    )
    assert plan["placement"]["resident"]["dense_core"] == "vram0"
    island = plan["placement"]["tiered"]["routed_experts"]["island"]
    assert island["node"] == "vram1"
    assert island["mode"] == "resident"
    assert island["policy"] == "static_hotlist"
    assert island["budget_gb"] == round(24 * 0.9, 1)
    assert "island_placement_requires_runtime_multi_gpu_support" in plan["risks"]
    assert validate_instance(plan) == []


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
