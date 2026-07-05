"""Experiment harness pure functions: log parsing, needle prompts, KV folding."""

from pathlib import Path

from frontier_bridge.bench.experiments import (
    build_needle_prompt,
    kv_per_1k_tokens_mb,
    parse_kv_size,
    parse_metal_limits,
)


def test_parse_metal_limits(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text(
        "ggml_metal_init: recommendedMaxWorkingSetSize  = 115343.36 MB\n"
    )
    assert parse_metal_limits(log)["recommended_max_working_set_gb"] == 112.6
    assert parse_metal_limits(tmp_path / "missing.log") == {
        "recommended_max_working_set_gb": None
    }


def test_parse_kv_size_sums_per_device_lines(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text(
        "llama_kv_cache_unified: Metal KV buffer size =  2048.00 MiB\n"
        "llama_kv_cache_unified: CPU KV buffer size =   512.50 MiB\n"
    )
    assert parse_kv_size(log) == 2560.5
    assert parse_kv_size(tmp_path / "missing.log") is None


def test_parse_kv_size_current_llamacpp_format(tmp_path: Path):
    """Current builds log a summary line (at -lv 5) plus per-device buffer lines
    that can read 0.00 on Metal; the summary must win over the buffer lines."""
    log = tmp_path / "server.log"
    log.write_text(
        "0.00.171.900 I llama_kv_cache:       MTL0 KV buffer size =     0.00 MiB\n"
        "0.00.171.902 I llama_kv_cache: size =  576.00 MiB (  4096 cells,  36 layers,"
        "  4/1 seqs), K (f16):  288.00 MiB, V (f16):  288.00 MiB\n"
    )
    assert parse_kv_size(log) == 576.0


def test_parse_kv_size_prefers_self_size(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text(
        "llama_kv_cache: KV self size  = 1024.00 MiB\n"
        "llama_kv_cache: CPU KV buffer size = 999.00 MiB\n"
    )
    assert parse_kv_size(log) == 1024.0


def test_build_needle_prompt_contains_needle_and_question():
    prompt = build_needle_prompt(4000)
    assert "BLUEHERON-42" in prompt
    assert prompt.rstrip().endswith("What is it exactly?")
    # Roughly sized: filler repeats scale with the token target.
    assert len(build_needle_prompt(20000)) > len(prompt) * 3


def test_kv_per_1k_tokens_folds_by_quant():
    records = [
        {"status": "stable", "ctx": 16384, "kv_size_mib": 1638.4, "kv_quant": "f16"},
        {"status": "stable", "ctx": 32768, "kv_size_mib": 3276.8, "kv_quant": "f16"},
        {"status": "stable", "ctx": 16384, "kv_size_mib": 819.2, "kv_quant": "q8_0"},
        # Unhealthy or unparsed rungs must not count.
        {"status": "load_timeout", "ctx": 65536, "kv_size_mib": 999999, "kv_quant": "f16"},
        {"status": "stable", "ctx": 65536, "kv_size_mib": None, "kv_quant": "f16"},
    ]
    folded = kv_per_1k_tokens_mb(records)
    assert folded == {"f16": 100.0, "q8_0": 50.0}


def test_kv_per_1k_tokens_empty():
    assert kv_per_1k_tokens_mb([]) == {}
