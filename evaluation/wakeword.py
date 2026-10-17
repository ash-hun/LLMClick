"""`evaluation_wakeword`: a wake word detector on recordings it never trained on and on hours of audio without the
phrase: the miss rate per noise condition, false accepts per hour, and both over every threshold."""

import math
from pathlib import Path
from typing import Any, ClassVar
from collections.abc import Iterable, Iterator

import numpy as np
from pydantic import Field, model_validator

from core.utils.files import atomic_path, directory_signature, read_json, read_jsonl, write_json, write_jsonl
from modeling.wakeword.stages import BUNDLE, LATE, LEAD, REFRACTORY, content, firings, rng_for, stream
from modeling.config import Keyed, keyed_identity, local_paths
from modeling.wakeword import detector, features
from evaluation.config import MeasureSettings
from core.config.schema import Section
from core.stage import Outputs, Stage
from modeling.tts import audio

TRACES = "traces.npz"
REPORT = "report.json"
STAGES = ("validate", "train")    # `model.checkpoint` values that name a stage of the experiment
CHUNK = 750 * features.HOP        # 60 s of ambient audio scored at once; chunks overlap by a window, so scores match
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)
Z = 1.96                          # 95% intervals
BY_TEXT = 20                      # texts of the negatives the report lists, most often fired first


class WakewordModel(Section):
    experiment: str | None = Field(default=None, description="A `wakeword` experiment directory (output/<name>-<hash>)")
    checkpoint: str = Field(default="validate", description="validate or train (a stage of the experiment), or a bundle "
                                                            "directory: wakeword.json beside the detector and the two feature models")

    @model_validator(mode="after")
    def _located(self) -> "WakewordModel":
        if self.checkpoint in STAGES and self.experiment is None:
            raise ValueError(f"model.checkpoint {self.checkpoint!r} needs model.experiment; or give a bundle directory")
        return self

    def bundle(self) -> Path:
        if self.checkpoint in STAGES:
            return Path(str(self.experiment)) / self.checkpoint / "checkpoint"
        return Path(self.checkpoint)

    def paths(self) -> list[str]:
        return [*([self.experiment] if self.experiment else []), str(self.bundle())]


class WakewordEvaluationData(Section):
    positive: list[Keyed] = Field(min_length=1, description="Recordings of the phrase (`audio`, optional `text`), ideally "
                                                            "by voices the detector never trained on")
    negative: list[Keyed] = Field(default_factory=list, description="Short recordings that must not wake it (near misses, "
                                                                    "commands); the report counts them per text")
    ambient: list[str] = Field(default_factory=list, description="Directories of WAVs that never say the phrase (speech, "
                                                                 "TV, music), hours of them: false accepts per hour")
    noise: list[str] = Field(default_factory=list, description="Directories of noise WAVs mixed under the recordings in "
                                                               "the noisy conditions; empty: Gaussian noise")

    def paths(self) -> list[str]:
        return [*local_paths(self.positive, audio.AUDIO_KEYS), *local_paths(self.negative, audio.AUDIO_KEYS),
                *self.ambient, *self.noise]


class WakewordEvaluationConfig(MeasureSettings):
    model: WakewordModel
    data: WakewordEvaluationData
    conditions: list[float | None] = Field(default_factory=lambda: [None, 10.0, 5.0], min_length=1,
                                           description="Speech-to-noise ratios in dB every recording is tested at; null: clean")
    threshold: float = Field(default=0.5, gt=0, lt=1, description="The operating point the headline numbers are read at")
    refractory: float = Field(default=REFRACTORY, gt=0, description="Seconds after a firing in which another one is not counted")
    targets: list[float] = Field(default_factory=lambda: [0.5, 0.1], description="False accepts per hour at which the "
                                                                                 "miss rate is read off the threshold sweep")

    def paths(self) -> list[str]:
        return [*super().paths(), *self.model.paths(), *self.data.paths()]


def label(snr: float | None) -> str:
    return "clean" if snr is None else f"snr{snr:g}"


def wilson(hits: int, total: int) -> list[float]:
    """Pure: the 95% Wilson interval of a share."""
    if not total:
        return [0.0, 1.0]
    share, spread = hits / total, Z * Z / total
    centre = (share + spread / 2) / (1 + spread)
    half = Z * math.sqrt(share * (1 - share) / total + spread / (4 * total)) / (1 + spread)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def poisson(events: int, hours: float) -> list[float]:
    """Pure: the 95% interval of a rate per hour from a count (Byar's approximation of the exact Poisson interval)."""
    if hours <= 0:
        return [0.0, math.inf]
    low = 0.0 if events == 0 else events * (1 - 1 / (9 * events) - Z / (3 * math.sqrt(events))) ** 3
    high = (events + 1) * (1 - 1 / (9 * (events + 1)) + Z / (3 * math.sqrt(events + 1))) ** 3
    return [low / hours, high / hours]


def hit(scores: np.ndarray, seconds: float, threshold: float, refractory: float) -> float | None:
    """Pure: seconds from the end of the phrase to the firing that heard it (negative: before its end), or None.
    The phrase starts LEAD seconds into its stream; a firing counts from its start to LATE seconds after its end."""
    for index in firings(scores, threshold, refractory):
        at = features.window_end(index) / features.RATE
        if LEAD <= at <= LEAD + seconds + LATE:
            return at - LEAD - seconds
    return None


def concatenated(paths: Iterable[str]) -> Iterator[np.ndarray]:
    """The ambient recordings as one stream, in chunks of CHUNK samples plus the window's overlap, without holding
    hours of audio in memory; as an always-on microphone hears them, one after another."""
    overlap, buffer = features.WINDOW - features.HOP, np.zeros(0, dtype=np.float32)
    for path in paths:
        buffer = np.concatenate([buffer, features.clip(Path(path))])
        while len(buffer) >= CHUNK + overlap:
            yield buffer[: CHUNK + overlap]
            buffer = buffer[CHUNK:]
    if len(buffer) >= features.WINDOW:
        yield buffer


class ClipsStage(Stage[WakewordEvaluationConfig]):
    """The recordings of every source found on disk, and the ambient and noise WAVs listed with their length."""
    name: ClassVar[str] = "clips"

    def identity(self) -> Any:
        data = self.config.data
        sources = [*data.positive, *data.negative]
        return [keyed_identity(sources), audio.signatures(sources),
                [directory_signature(Path(path)) for path in [*data.ambient, *data.noise]]]

    def rows(self, sources: list[Keyed]) -> list[dict[str, Any]]:
        rows = []
        for source in sources:
            for number, row, path in audio.recordings(source):
                if path is None:
                    raise ValueError(f"{source.name} row {number}: no audio file at {row.get('audio')!r} under {audio.audio_root(source)}")
                rows.append({"audio": str(path), "text": row.get("text"), "seconds": audio.seconds(path)})
        return rows

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        data = self.config.data
        files = {"positive": self.rows(data.positive), "negative": self.rows(data.negative)}
        for name, directories in (("ambient", data.ambient), ("noise", data.noise)):
            found = sorted(str(path.resolve()) for directory in directories for path in Path(directory).rglob("*.wav"))
            if directories and not found:
                raise ValueError(f"no .wav files under {directories}")
            files[name] = [{"audio": path, "seconds": audio.seconds(Path(path))} for path in found]
        for name, rows in files.items():
            write_jsonl(workdir / f"{name}.jsonl", rows)
        return {**{name: str(workdir / f"{name}.jsonl") for name in files},
                "counts": {name: len(rows) for name, rows in files.items()},
                "ambient_hours": sum(row["seconds"] for row in files["ambient"]) / 3600}


class TraceStage(Stage[WakewordEvaluationConfig]):
    """The detector's score every 80 ms: on each recording in a stream of its own under every condition (as
    `validate` builds them), and on the ambient audio as one long stream. Thresholds are the report's business, so
    a new operating point reads these scores again instead of listening again. Its scope shares the work with any
    evaluation of the same detector on the same recordings."""
    name: ClassVar[str] = "traces"
    scope: ClassVar[str] = "wakeword-traces"
    requires: ClassVar[tuple[str, ...]] = ("clips",)
    sections: ClassVar[tuple[str, ...]] = ("seed", "conditions")

    def identity(self) -> Any:
        return [super().identity(), directory_signature(self.config.model.bundle())]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        bundle, clips = self.config.model.bundle(), inputs["clips"]
        if not (bundle / BUNDLE).is_file():
            raise FileNotFoundError(f"{bundle} has no {BUNDLE}: not a wakeword checkpoint")
        meta = read_json(bundle / BUNDLE)
        self.progress.update(0, None, "loading the detector")
        score = detector.Scorer(bundle, meta["model"])
        noise = [features.clip(Path(row["audio"])) for row in read_jsonl(Path(clips["noise"]))]
        traces: dict[str, Any] = {}
        recorded = [(kind, read_jsonl(Path(clips[kind]))) for kind in ("positive", "negative")]
        total = sum(len(rows) for _, rows in recorded) * len(self.config.conditions)
        done = 0
        for condition, snr in enumerate(self.config.conditions):
            for kind, rows in recorded:
                for index, row in enumerate(rows):
                    self.progress.update(done, total, f"{kind} recordings, {label(snr)}")
                    rng = rng_for(self.config.seed, "evaluation", kind, snr, content(row["audio"]))
                    traces[f"{kind}/{condition}/{index}"] = score(stream([features.clip(Path(row["audio"]))], rng, noise, snr))
                    done += 1
        ambient = read_jsonl(Path(clips["ambient"]))
        if ambient:
            chunks = []
            for index, chunk in enumerate(concatenated(row["audio"] for row in ambient)):
                self.progress.update(index, math.ceil(clips["ambient_hours"] * 3600 * features.RATE / CHUNK), "ambient audio")
                chunks.append(score(chunk))
            traces["ambient"] = np.concatenate(chunks)
        with atomic_path(workdir / TRACES) as temporary, temporary.open("wb") as handle:
            np.savez(handle, **traces)
        return {"traces": str(workdir / TRACES), "phrase": meta["phrase"], "bundle": str(bundle)}


class WakewordReport(Stage[WakewordEvaluationConfig]):
    """The scores read at every threshold: misses per condition, false accepts per hour of ambient audio and per
    text of the negatives; the operating point, the sweep, and the miss rate where false accepts reach each target."""
    name: ClassVar[str] = "report"
    requires: ClassVar[tuple[str, ...]] = ("clips", "traces")
    sections: ClassVar[tuple[str, ...]] = ("threshold", "refractory", "targets")

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        config, clips = self.config, inputs["clips"]
        traces = np.load(inputs["traces"]["traces"])
        positives, negatives = read_jsonl(Path(clips["positive"])), read_jsonl(Path(clips["negative"]))
        hours = clips["ambient_hours"]
        names = [label(snr) for snr in config.conditions]

        def misses(condition: int, threshold: float) -> list[float | None]:
            return [hit(traces[f"positive/{condition}/{index}"], row["seconds"], threshold, config.refractory)
                    for index, row in enumerate(positives)]

        def fired(condition: int, threshold: float) -> list[bool]:
            return [bool(firings(traces[f"negative/{condition}/{index}"], threshold, config.refractory))
                    for index in range(len(negatives))]

        def per_hour(threshold: float) -> tuple[int, float | None]:
            if "ambient" not in traces:
                return 0, None
            events = len(firings(traces["ambient"], threshold, config.refractory))
            return events, events / hours

        sweep = []
        for threshold in THRESHOLDS.tolist():
            frr = {name: sum(late is None for late in misses(condition, threshold)) / len(positives)
                   for condition, name in enumerate(names)}
            events, rate = per_hour(threshold)
            sweep.append({"threshold": threshold, "frr": frr, "fa_per_hour": rate, "fa_events": events,
                          "negatives_fired": {name: sum(fired(condition, threshold)) / len(negatives) if negatives else None
                                              for condition, name in enumerate(names)}})

        at_targets: dict[str, Any] = {}
        for target in config.targets:
            # ponytail: the most sensitive swept threshold within the target, not interpolated between thresholds;
            # interpolate (Picovoice brackets the target) when 0.01 steps prove too coarse
            within = [point for point in sweep if point["fa_per_hour"] is not None and point["fa_per_hour"] <= target]
            at_targets[f"{target:g}"] = ({"threshold": within[0]["threshold"], "frr": within[0]["frr"]} if within else None)

        operating: dict[str, Any] = {"threshold": config.threshold, "conditions": {}}
        for condition, name in enumerate(names):
            lates = misses(condition, config.threshold)
            heard = [late for late in lates if late is not None]
            missed = len(lates) - len(heard)
            operating["conditions"][name] = {
                "frr": missed / len(lates), "frr_interval": wilson(missed, len(lates)), "missed": missed, "positives": len(lates),
                "latency_median": float(np.median(heard)) if heard else None,
                "negatives_fired": (sum(fired(condition, config.threshold)) / len(negatives)) if negatives else None}
        events, rate = per_hour(config.threshold)
        operating["ambient"] = {"hours": hours, "events": events, "fa_per_hour": rate,
                                "fa_interval": poisson(events, hours) if rate is not None else None}
        by_text: dict[str, list[int]] = {}
        for row, was in zip(negatives, fired(0, config.threshold), strict=True):
            counts = by_text.setdefault(str(row.get("text")), [0, 0])
            counts[0], counts[1] = counts[0] + int(was), counts[1] + 1
        operating["negatives_by_text"] = [{"text": text, "fired": hit_, "total": total} for text, (hit_, total) in
                                          sorted(by_text.items(), key=lambda item: (-item[1][0] / item[1][1], item[0]))][:BY_TEXT]

        scores: dict[str, float | None] = {f"frr_{name}": entry["frr"] for name, entry in operating["conditions"].items()}
        scores["fa_per_hour"] = rate
        for key, point in at_targets.items():
            for name in names:
                scores[f"frr_{name}_at_{key}_fa_per_hour"] = point["frr"][name] if point else None
        report = {"model": {"bundle": inputs["traces"]["bundle"], "phrase": inputs["traces"]["phrase"],
                            "experiment": config.model.experiment, "checkpoint": config.model.checkpoint},
                  "settings": {"conditions": names, "refractory": config.refractory, "seed": config.seed,
                               "noise": "files" if clips["counts"]["noise"] else "gaussian"},
                  "data": clips["counts"] | {"ambient_hours": hours},
                  "scores": scores, "operating_point": operating, "at_targets": at_targets, "sweep": sweep}
        write_json(workdir / REPORT, report)
        return {"report": str(workdir / REPORT), "scores": scores}
