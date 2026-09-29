from jeff import domain


def test_families_never_straddle_the_split() -> None:
    rows = [{"id": f"r{i}", "family": f"f{i % 200}", "source": {}} for i in range(2000)]
    parts = domain.split(rows, seed=1)  # type: ignore[arg-type]
    families = {name: {r["family"] for r in found} for name, found in parts.items()}
    assert not (families["train"] & families["dev"]) and not (families["train"] & families["calibration"])
    assert not (families["dev"] & families["calibration"])
    assert 0.05 < len(parts["dev"]) / 2000 < 0.16 and len(parts["calibration"]) > 0
