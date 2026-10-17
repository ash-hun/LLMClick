"""The stages of `wakeword`: recordings -> noisy windows as embeddings -> a small detector -> a streaming test."""

import shutil
import hashlib
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from modeling.tuning.stages import held_out_positions, read_rows, write_rows
from modeling.wakeword.config import AugmentConfig, WakewordConfig
from core.utils.files import directory_signature, read_json, write_json
from modeling.stages import TrainStage, ValidateStage
from modeling.tts.stages import audio_root, located
from modeling.tuning.sources import SOURCES
from modeling.config import Keyed, keyed_identity
from modeling.wakeword import features, detector
from modeling.tracker import History
from core.stage import Outputs, Stage
from modeling.tts.asr import words

FEATURES = "features.npz"
SUMMARY = "summary.json"
BUNDLE = "wakeword.json"
LEAD, GAP = 2.0, 0.6      # seconds of noise before a test stream's speech (a full window, as an always-on microphone has), and between the phrase and the command
LATE = 1.0                # seconds after the phrase's end within which firing still counts as hearing it
REFRACTORY = 1.0          # seconds after firing in which the detector is not counted again
FOLLOW = 0.6              # seconds of other speech after the phrase in a `followed` window, up to twice this
MARGIN = 0.2              # seconds between the latest a positive window ends and the earliest a `followed` one does


def rng_for(seed: int, *parts: Any) -> np.random.Generator:
    """A generator that depends only on the seed and the item, so a window is the same whenever it is made."""
    return np.random.default_rng([seed, int(hashlib.sha256(repr(parts).encode()).hexdigest()[:15], 16)])


def backdrop(rng: np.random.Generator, background: list[np.ndarray], length: int) -> np.ndarray:
    """`length` samples of noise: a random stretch of a background recording, looped if short, else Gaussian."""
    if not background:
        return rng.standard_normal(length).astype(np.float32)
    noise = background[int(rng.integers(len(background)))]
    start = int(rng.integers(len(noise)))
    return np.asarray(np.resize(np.roll(noise, -start), length), dtype=np.float32)


def mixed(speech: np.ndarray, noise: np.ndarray, snr: float | None) -> np.ndarray:
    """Pure: `noise` under `speech` at `snr` dB (by the speech's power); None leaves the speech clean."""
    if snr is None:
        return speech
    power = float(np.mean(speech[np.abs(speech) > 0] ** 2)) if np.any(speech) else 1e-4
    scale = np.sqrt(power / (10 ** (snr / 10)) / max(float(np.mean(noise ** 2)), 1e-12))
    return np.asarray(speech + noise * scale, dtype=np.float32)


def window(samples: np.ndarray, rng: np.random.Generator, augment: AugmentConfig, background: list[np.ndarray],
           crop: bool) -> np.ndarray:
    """One training window of features.WINDOW samples: the recording (cut short at a random point when `crop`)
    ending a random `tail` before the window's end, over noise, at a random gain."""
    if crop and len(samples):
        samples = samples[: int(rng.integers(min(len(samples), features.RATE // 2), len(samples) + 1))]
    end = features.WINDOW - int(rng.uniform(0, augment.tail) * features.RATE)
    part = samples[-end:]
    speech = np.zeros(features.WINDOW, dtype=np.float32)
    speech[end - len(part):end] = part
    noisy = mixed(speech, backdrop(rng, background, features.WINDOW), float(rng.uniform(*augment.snr)))
    gain = 10 ** (float(rng.uniform(*augment.gain)) / 20) / max(1.0, float(np.abs(noisy).max()))
    return np.asarray(np.clip(noisy * gain, -1, 1), dtype=np.float32)


def then(phrase: np.ndarray, other: np.ndarray, rng: np.random.Generator, augment: AugmentConfig) -> np.ndarray:
    """The phrase, a pause and up to 2 x FOLLOW seconds of another recording (looped if short), cut at a random point
    at least `augment.tail` + MARGIN seconds after the phrase: a window ending there heard the phrase a while ago,
    which is no reason to fire again, whether the user is silent or already giving the command."""
    pause = np.zeros(int(rng.uniform(0.1, GAP) * features.RATE), dtype=np.float32)
    after = np.resize(other, int(rng.uniform(FOLLOW, 2 * FOLLOW) * features.RATE)) if len(other) else np.zeros(0)
    following = np.concatenate([pause, after.astype(np.float32)])
    shortest = min(len(following), int((augment.tail + MARGIN) * features.RATE))
    return np.concatenate([phrase, following[: int(rng.integers(shortest, len(following) + 1))]])


def stream(parts: list[np.ndarray], rng: np.random.Generator, background: list[np.ndarray], snr: float | None) -> np.ndarray:
    """Pure but for `rng`: the recordings one after another, with LEAD seconds before, GAP between and after, over noise."""
    gap = np.zeros(int(GAP * features.RATE), dtype=np.float32)
    pieces = [np.zeros(int(LEAD * features.RATE), dtype=np.float32)]
    for part in parts:
        pieces += [part, gap]
    speech = np.concatenate(pieces)
    return np.asarray(np.clip(mixed(speech, backdrop(rng, background, len(speech)), snr), -1, 1), dtype=np.float32)


def firings(scores: np.ndarray, threshold: float) -> list[int]:
    """Pure: the windows at which the detector fires, at most once per REFRACTORY seconds."""
    quiet = int(REFRACTORY * features.RATE / features.HOP)
    fired: list[int] = []
    for index, score in enumerate(scores):
        if score >= threshold and (not fired or index - fired[-1] > quiet):
            fired.append(index)
    return fired


def contains(said: list[str], phrase: list[str]) -> bool:
    """Pure: whether the phrase's words occur together in `said`."""
    return any(said[start:start + len(phrase)] == phrase for start in range(len(said) - len(phrase) + 1))


class ClipsStage(Stage[WakewordConfig]):
    """Every recording of both classes found on disk, the texts checked (a positive says the phrase, a negative never
    does), and the validation share of each class held out."""
    name: ClassVar[str] = "data"

    def identity(self) -> Any:
        data = self.config.data
        sources = [*data.positive, *data.negative]
        return [self.config.seed, self.config.phrase, keyed_identity(sources), data.validation,
                [directory_signature(root) if (root := audio_root(source)) else None for source in sources],
                [directory_signature(Path(path)) for path in data.background]]

    def rows(self, sources: list[Keyed], label: int) -> list[dict[str, Any]]:
        phrase = words(self.config.phrase)
        rows: list[dict[str, Any]] = []
        for index, source in enumerate(sources):
            root = audio_root(source)
            for number, row in enumerate(SOURCES.get(source.name)(source.params)):
                path = located(row.get("audio"), root)
                if path is None:
                    raise ValueError(f"{source.name} row {number}: no audio file at {row.get('audio')!r} under {root}")
                said = words(str(row.get("text") or ""))
                if label and said and said != phrase:
                    raise ValueError(f"{source.name} row {number} says {row['text']!r}, not the phrase {self.config.phrase!r}")
                if not label and contains(said, phrase):
                    raise ValueError(f"negative {source.name} row {number} says the phrase: {row['text']!r}")
                rows.append({"audio": str(path), "label": label, "source": index, "text": row.get("text")})
        return rows

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        data = self.config.data
        split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
        for label, sources in ((1, data.positive), (0, data.negative)):
            self.progress.update(1 - label, 2, "reading recordings")
            rows = self.rows(sources, label)
            if len(rows) < 2:
                raise ValueError(f"{len(rows)} {'positive' if label else 'negative'} recordings cannot be split for validation")
            held_out = held_out_positions(len(rows), self.config.seed + label, data.validation)
            split["train"] += [row for i, row in enumerate(rows) if i not in held_out]
            split["validation"] += [row for i, row in enumerate(rows) if i in held_out]
        background = sorted(str(path.resolve()) for directory in data.background for path in Path(directory).rglob("*.wav"))
        if data.background and not background:
            raise ValueError(f"no .wav files under {data.background}")
        for name, fold in split.items():
            write_rows(workdir / f"{name}.jsonl", fold)
        write_rows(workdir / "background.jsonl", [{"audio": path} for path in background])
        counts = {name: {"positive": sum(row["label"] for row in fold), "negative": sum(1 - row["label"] for row in fold)}
                  for name, fold in split.items()}
        return {**{name: str(workdir / f"{name}.jsonl") for name in split}, "background": str(workdir / "background.jsonl"),
                "rows": counts, "background_files": len(background)}


def background_of(inputs: dict[str, Outputs]) -> list[np.ndarray]:
    # ponytail: every background file is held in memory at 16 kHz; stream from disk when the noise set reaches gigabytes
    return [features.clip(Path(row["audio"])) for row in read_rows(Path(inputs["data"]["background"]))]


class WindowsStage(Stage[WakewordConfig]):
    """Each training recording cut into `augment.rounds` noisy windows (positives whole, negatives also cut short,
    so a half-said phrase is a negative), the phrase followed by other speech as negatives, and windows of noise
    alone, all turned into embeddings. Every window's randomness comes from the seed and its recording, so a rerun
    makes the same file."""
    name: ClassVar[str] = "features"
    requires: ClassVar[tuple[str, ...]] = ("data",)
    sections: ClassVar[tuple[str, ...]] = ("seed", "augment")

    def identity(self) -> Any:
        return [super().identity(), features.MODELS]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        augment, seed = self.config.augment, self.config.seed
        self.progress.update(0, None, "loading the feature models")
        embed = features.Embedder(features.ensure(workdir / "models"))
        background = background_of(inputs)
        rows = read_rows(Path(inputs["data"]["train"]))
        others = [row["audio"] for row in rows if not row["label"]]
        windows: list[np.ndarray] = []
        labels: list[int] = []
        for done, row in enumerate(rows):
            self.progress.update(done, len(rows), "embedding windows")
            samples = features.clip(Path(row["audio"]))
            for round_ in range(augment.rounds):
                rng = rng_for(seed, row["audio"], round_)
                windows.append(embed(window(samples, rng, augment, background, crop=not row["label"]))[-features.FRAMES:])
                labels.append(int(row["label"]))
            for round_ in range(augment.followed if row["label"] else 0):
                rng = rng_for(seed, row["audio"], "followed", round_)
                other = features.clip(Path(others[int(rng.integers(len(others)))]))
                windows.append(embed(window(then(samples, other, rng, augment), rng, augment, background, crop=False))[-features.FRAMES:])
                labels.append(0)
        for index in range(augment.noise):
            rng = rng_for(seed, "noise", index)
            windows.append(embed(window(np.zeros(0, dtype=np.float32), rng, augment, background, crop=False))[-features.FRAMES:])
            labels.append(0)
        self.progress.update(len(rows), len(rows))
        temporary = workdir / (FEATURES + ".tmp.npz")
        np.savez(temporary, x=np.stack(windows).astype(np.float32), y=np.asarray(labels, dtype=np.float32))
        temporary.replace(workdir / FEATURES)
        counts = {"positive": int(sum(labels)), "negative": len(labels) - int(sum(labels))}
        return {"features": str(workdir / FEATURES), "models": str(workdir / "models"), "windows": counts}


class FitDetectorStage(TrainStage[WakewordConfig]):
    """openWakeWord's DNN head trained on the windows and exported as the ONNX file its runtime loads, next to the
    two feature models. Training is seeded on the CPU, so a rerun after a crash trains the same detector again."""
    requires: ClassVar[tuple[str, ...]] = ("features",)
    sections: ClassVar[tuple[str, ...]] = ("seed", "phrase", "training")

    def history(self, outputs: Outputs) -> History:
        return [(int(step), {"loss": float(loss)}) for step, loss in outputs.get("history", [])]

    def train(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        checkpoint = workdir / "checkpoint"
        if not (checkpoint / SUMMARY).is_file():
            self.progress.update(0, self.config.training.steps, "training the detector")
            data = np.load(inputs["features"]["features"])
            staging = workdir / "checkpoint.tmp"
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir()
            model_file = f"{self.config.slug}.onnx"
            history = detector.fit(data["x"], data["y"], self.config.training, self.config.seed, staging / model_file,
                                   lambda step: self.progress.update(step, self.config.training.steps, "training the detector"))
            for name in features.MODELS:
                shutil.copyfile(Path(inputs["features"]["models"]) / name, staging / name)
            write_json(staging / BUNDLE, {"phrase": self.config.phrase, "model": model_file, "frames": features.FRAMES,
                                          "sample_rate": features.RATE, "features": {"release": features.RELEASE, **features.MODELS}})
            write_json(staging / SUMMARY, {"model": model_file, "windows": inputs["features"]["windows"], "history": history})
            staging.rename(checkpoint)
        return {"checkpoint": str(checkpoint), **read_json(checkpoint / SUMMARY)}


class StreamTestStage(ValidateStage[WakewordConfig]):
    """The exported detector over held-out recordings in noise, scored every 80 ms as the runtime would. Metrics:
    `recall` (phrase streams it fires on), `false_accepts` (other streams it fires on), and `scenario`: phrase then
    a command in one stream, where the detector must fire once, on the phrase, and not on the command."""
    requires: ClassVar[tuple[str, ...]] = ("train", "data")
    sections: ClassVar[tuple[str, ...]] = ("validation", "seed")

    def measure(self, workdir: Path, inputs: dict[str, Outputs]) -> dict[str, float]:
        settings, seed = self.config.validation, self.config.seed
        bundle = Path(inputs["train"]["checkpoint"])
        score = detector.Scorer(bundle, read_json(bundle / BUNDLE)["model"])
        background = background_of(inputs)
        rows = read_rows(Path(inputs["data"]["validation"]))
        positives = [row for row in rows if row["label"]]
        negatives = [row for row in rows if not row["label"]]
        items: list[dict[str, Any]] = []
        for done, row in enumerate(rows):
            self.progress.update(done, len(rows) + len(positives), "listening for the phrase")
            fired = firings(score(stream([features.clip(Path(row["audio"]))], rng_for(seed, "test", row["audio"]),
                                         background, settings.snr)), settings.threshold)
            items.append({"test": "single", "audio": row["audio"], "label": row["label"], "fired": len(fired)})
        for done, row in enumerate(positives):
            self.progress.update(len(rows) + done, len(rows) + len(positives), "phrase, then a command")
            phrase = features.clip(Path(row["audio"]))
            command = negatives[done % len(negatives)]
            scores = score(stream([phrase, features.clip(Path(command["audio"]))], rng_for(seed, "scenario", row["audio"]),
                                  background, settings.snr))
            fired_at = [features.window_end(index) / features.RATE for index in firings(scores, settings.threshold)]
            said = (LEAD, LEAD + len(phrase) / features.RATE)
            passed = len(fired_at) == 1 and said[0] <= fired_at[0] <= said[1] + LATE
            items.append({"test": "scenario", "audio": row["audio"], "command": command["audio"], "fired_at": fired_at,
                          "phrase_at": said, "passed": passed})
        self.progress.update(len(rows) + len(positives), len(rows) + len(positives))
        write_rows(workdir / "items.jsonl", items)
        singles = [item for item in items if item["test"] == "single"]
        return {"recall": float(np.mean([item["fired"] > 0 for item in singles if item["label"]])),
                "false_accepts": float(np.mean([item["fired"] > 0 for item in singles if not item["label"]])),
                "scenario": float(np.mean([item["passed"] for item in items if item["test"] == "scenario"]))}
