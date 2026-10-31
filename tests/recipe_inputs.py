"""Configs and input files the recipe tests share: a tiny run of each recipe and the recordings it reads."""

import json
from typing import Any
from pathlib import Path

from tests.speech_fakes import melody, tone

RECIPES = {  # recipe -> (sample file, method section, metric the validate stage must report)
    "llm_sft": ("llm_sft", {}, "loss"),
    "llm_instruction": ("llm_instruction", {"system": "Add."}, "perplexity"),
    "llm_dpo": ("llm_dpo", {"beta": 0.1}, "accuracy"),
    "llm_grpo": ("llm_grpo", {"group_size": 3, "max_new_tokens": 3, "reward": {"name": "contains"}}, "reward"),
    "embedding_contrastive": ("embedding_contrastive", {"temperature": 0.1}, "mrr"),
}
SFT_ROWS = [json.loads(line) for line in Path("samples/llm_sft.jsonl").read_text().splitlines()][:12]
DECISION_ROWS = [json.loads(line) for line in Path("samples/llm_decision.jsonl").read_text().splitlines()]
# "Jarvis" then "wake up" as two notes; the negatives are other notes, and the first note alone (the name alone)
NAME, WAKE, OTHERS = 880, 1320, [300, 440, 1500, 2200]
TEXTS = ["Good evening, sir.", "All systems are online.", "Shall I run the diagnostics?", "The suit is ready.",
         "Accessing the alarm settings.", "You have three new messages."]


def raw(recipe: str, model: Path, out: Path, **training: Any) -> dict[str, Any]:
    sample, method, _ = RECIPES[recipe]
    return {
        "pipeline": {"recipe": recipe, "name": "tiny", "seed": 3, "output_dir": str(out), "device": "cpu"},
        "model": {"architecture": "bi_encoder" if recipe.startswith("embedding") else "transformer", "name": str(model)},
        "data": {"sources": [{"name": "local_jsonl", "path": f"samples/{sample}.jsonl"}], "validation": 0.2},
        "method": dict(method),  # a copy: tests change it, and RECIPES is shared
        "training": {"lr": 1e-3, "batch_size": 4, "max_length": 64, "max_steps": 3, **training},
    }


def decision_raw(recipe: str, model: Path, out: Path, head: str, method: dict[str, Any] | None = None,
                 **model_keys: Any) -> dict[str, Any]:
    return {
        "pipeline": {"recipe": recipe, "name": "tiny", "seed": 3, "output_dir": str(out), "device": "cpu"},
        "model": {"architecture": "transformer", "name": str(model), "head": head, **model_keys},
        "data": {"sources": [{"name": "local_jsonl", "path": "samples/llm_decision.jsonl"}], "validation": 0.2},
        "method": {"calibration": 0.2, "max_think": 6, **(method or {})},
        "training": {"lr": 1e-3, "batch_size": 4, "max_length": 250, "max_steps": 3},
    }


def wakeword_recordings(root: Path) -> Path:
    """Two sets of rows.jsonl beside their wavs, as data_speech writes them; outside output/, which the signature of
    their directory would otherwise include."""
    root = root / "data"
    positive, negative = [], []
    for index in range(10):
        melody(root / "wake" / f"{index}.wav", [(NAME, 0.4), (WAKE, 0.4 + 0.05 * index)])
        positive.append({"audio": f"wake/{index}.wav", "text": "Jarvis, wake up."})
        if index % 3:
            melody(root / "other" / f"{index}.wav", [(OTHERS[index % len(OTHERS)], 1.0 + 0.1 * index)])
            negative.append({"audio": f"other/{index}.wav", "text": "Please search the web."})
        else:
            melody(root / "other" / f"{index}.wav", [(NAME, 0.5 + 0.05 * index)])
            negative.append({"audio": f"other/{index}.wav", "text": "Jarvis."})
    for name, rows in (("positive", positive), ("negative", negative)):
        (root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return root


def wakeword_config(root: Path, **keys: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "pipeline": {"recipe": "wakeword", "name": "wake", "seed": 7, "output_dir": str(root.parent / "output")},
        "phrase": "Jarvis, wake up",
        "data": {"positive": [{"name": "local_jsonl", "path": str(root / "positive.jsonl")}],
                 "negative": [{"name": "local_jsonl", "path": str(root / "negative.jsonl")}], "validation": 0.3},
        "augment": {"rounds": 2, "noise": 10, "snr": [20, 30]},
        "training": {"steps": 150, "batch_size": 32},
        "validation": {"min": {"recall": 1.0, "scenario": 1.0}, "max": {"false_accepts": 0.0}},
    }
    return {**base, **keys}


def tts_recordings(root: Path) -> Path:
    """An LJSpeech-style set: wavs/<id>.wav and a metadata.csv of `<id>|text` lines."""
    for index, text in enumerate(TEXTS):
        tone(root / "wavs" / f"utt{index}.wav", 0.2 * len(text.split()))
    (root / "metadata.csv").write_text("".join(f"utt{index}|{text}\n" for index, text in enumerate(TEXTS)))
    (root / "start.ckpt").write_text("pretrained")
    return root


def tts_config(root: Path, **keys: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "pipeline": {"recipe": "tts_piper", "name": "voice", "seed": 5, "output_dir": str(root / "output"), "device": "cpu"},
        "model": {"espeak_voice": "en-gb-x-rp", "init": {"name": str(root / "start.ckpt")}},
        "data": {"sources": [{"name": "delimited", "path": str(root / "metadata.csv"), "audio_dir": str(root / "wavs")}],
                 "validation": 0.34},
        "training": {"epochs": 3, "batch_size": 2, "args": {"model.learning_rate": 1.0e-4}},
        "validation": {"max": {"wer": 0.1}},
    }
    return {**base, **keys}
