"""Expert-sized SSD read microbenchmark (small real I/O, bounded)."""

from frontier_bridge.bench.streamspike import expert_read_bench, stream_read_gbps


def test_expert_read_bench_measures_real_io(tmp_path):
    records = expert_read_bench(
        [1, 2], file_gb=0.05, reads_per_size=4, directory=str(tmp_path)
    )
    assert [r["chunk_mb"] for r in records] == [1, 2]
    for record in records:
        assert record["gbps"] is None or record["gbps"] > 0
        assert record["reads"] == 4
        assert record["method"] == "uncached_random_chunk_reads_python"


def test_stream_read_gbps_takes_worst_measured():
    records = [
        {"chunk_mb": 4, "gbps": 5.4},
        {"chunk_mb": 16, "gbps": 7.6},
        {"chunk_mb": 64, "gbps": None},
    ]
    assert stream_read_gbps(records) == 5.4
    assert stream_read_gbps([{"gbps": None}]) is None
