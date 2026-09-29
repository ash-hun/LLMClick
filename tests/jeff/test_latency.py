from jeff import latency


def test_sample_is_even_across_suites_and_fixed() -> None:
    rows = [{"id": f"{s}{i}", "suite": s} for s in ("A", "B", "C", "D") for i in range(100)]
    first, second = latency.sample(rows, 40, seed=1), latency.sample(rows, 40, seed=1)
    assert first == second
    assert sorted({r["suite"] for r in first}) == ["A", "B", "C", "D"]
    assert all(sum(r["suite"] == s for r in first) == 10 for s in "ABCD")


def test_summary_statistics() -> None:
    summary = latency.summarise([float(ms) for ms in range(1, 101)])
    assert summary["median_ms"] == 50.5 and summary["p95_ms"] == 95.0
    assert summary["min_ms"] == 1.0 and summary["max_ms"] == 100.0
