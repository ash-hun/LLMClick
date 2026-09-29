import sys
from pathlib import Path

import pytest

from jeff import evaluate
from jeff.train import loss_stalled, prune_finished, selection_gate


def summary(ece: float, brier: float, accuracy: float = 0.5) -> dict[str, object]:
    return {"count": 10, "accuracy": accuracy, "ece": ece, "brier": brier, "nll": 1.0,
            "reliability": [], "zero_probability_count": 0}


def test_gate_without_reference_is_always_eligible() -> None:
    assert selection_gate(summary(0.9, 0.9), None) is True


def test_gate_with_reference_uses_calibration_margin() -> None:
    assert selection_gate(summary(0.01, 0.1), summary(0.05, 0.2)) is True
    assert selection_gate(summary(0.5, 0.9), summary(0.05, 0.2)) is False


def parse(argv: list[str]) -> evaluate.Arguments:
    return evaluate.parse_arguments(argv)


def test_model_kwargs_passes_base_and_revision() -> None:
    args = parse(["--data", "x.jsonl", "--output", "o.json", "--local",
                  "--base-model", "Qwen/Qwen3.5-0.8B", "--revision", "2fc06364715b967f1860aea9cf38778875588b17"])
    assert evaluate.model_kwargs(args) == {"base_model": "Qwen/Qwen3.5-0.8B",
                                           "revision": "2fc06364715b967f1860aea9cf38778875588b17"}


def test_model_kwargs_empty_by_default() -> None:
    assert evaluate.model_kwargs(parse(["--data", "x.jsonl", "--output", "o.json", "--local"])) == {}


@pytest.mark.parametrize("argv", [
    ["--local", "--base-model", "Qwen/Qwen3.5-0.8B"],                               # revision missing
    ["--local", "--checkpoint", "c", "--base-model", "m", "--revision", "r" * 40],  # checkpoint already names its base
    ["--local", "--checkpoint", "c", "--calibration", "cal.jsonl"],                 # trained checkpoints carry their own temperature
    ["--local", "--temperature", "1", "--calibration", "cal.jsonl"],
    ["--predictions", "p.jsonl", "--calibration", "cal.jsonl"],
])
def test_invalid_combinations_error(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse(["--data", "x.jsonl", "--output", "o.json", *argv])


def test_loss_stall_counts_evaluations_since_the_lowest_loss_from_epoch_two() -> None:
    rising = [(40, 0.50), (80, 0.40), (120, 0.41), (160, 0.42), (200, 0.43)]  # lowest at 80, then three evaluations above it
    assert not loss_stalled(rising, patience=None, first_active_step=1)       # off unless asked for
    assert not loss_stalled([], patience=3, first_active_step=1)
    assert loss_stalled(rising, patience=3, first_active_step=161)            # 200 is in epoch 2: stop
    assert not loss_stalled(rising, patience=3, first_active_step=201)        # still epoch 1: never stop
    assert not loss_stalled(rising, patience=4, first_active_step=1)          # only three evaluations since the lowest
    noisy = [(40, 0.50), (80, 0.40), (120, 0.45), (160, 0.39), (200, 0.41)]   # a rise, then a new lowest at 160
    assert not loss_stalled(noisy, patience=2, first_active_step=1)


def test_prune_keeps_only_the_selected_checkpoint(tmp_path) -> None:
    for step in ("step-00040", "step-00080", "step-00120"):
        (tmp_path / step).mkdir()
        (tmp_path / step / "model.safetensors").write_bytes(b"x" * 10)
    (tmp_path / "resume.pt").write_bytes(b"y" * 100)
    (tmp_path / "selected").symlink_to("step-00080", target_is_directory=True)
    (tmp_path / "final").symlink_to("step-00120", target_is_directory=True)
    assert prune_finished(tmp_path) == 110
    assert sorted(p.name for p in tmp_path.iterdir()) == ["final", "selected", "step-00080", "step-00120"]
    assert (tmp_path / "selected" / "model.safetensors").exists()


def test_selection_prefers_the_lowest_development_loss() -> None:
    from jeff.evaluate import selection_key
    noisy_accuracy = ({"nll": 0.40}, {"accuracy": 0.872}, 720)
    lower_loss = ({"nll": 0.35}, {"accuracy": 0.868}, 877)
    assert selection_key(*lower_loss) < selection_key(*noisy_accuracy)  # type: ignore[arg-type]
    tie = ({"nll": 0.35}, {"accuracy": 0.870}, 900)
    assert selection_key(*tie) < selection_key(*lower_loss)  # type: ignore[arg-type]  # same loss: higher accuracy wins


def test_initial_identity_fingerprints_single_file_checkpoints(tmp_path) -> None:
    import json
    from jeff.train import initial_identity
    (tmp_path / "decision_config.json").write_text(json.dumps({"format_version": 1, "base_model": "m", "revision": "r", "temperature": 1.2}))
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "readout.safetensors").write_bytes(b"readout")
    identity = initial_identity(str(tmp_path), "m", "r")
    assert sorted(identity["files_sha256"]) == ["decision_config.json", "model.safetensors", "readout.safetensors"]
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="no backbone weights"):
        initial_identity(str(tmp_path), "m", "r")
