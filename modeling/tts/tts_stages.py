"""The stages of the TTS recipes: recordings -> a voice (Piper fine-tuned, or Qwen3-TTS cloning one recording) -> a
listening test."""

import shutil
import statistics
from pathlib import Path
from typing import Any, ClassVar
from collections.abc import Callable

from core.file_store import atomic_path, read_json, read_jsonl, staged_directory, write_json, write_jsonl
from modeling.tts.tts_config import CHECKPOINT_FILE, PiperConfig, QwenConfig, VoiceConfig
from modeling.modeling_stages import TrainStage, ValidateStage, held_out_positions
from modeling.tts.quality_rater import METRICS, Rater, Speaker
from core.device_selector import resolve_device
from modeling.tts import audio_file, piper_process, qwen_process
from modeling.tts.asr_listener import Listener, wer
from modeling.modeling_config import keyed_identity
from core.stage_base import Outputs, Stage

VOICE = "voice.onnx"  # with VOICE + ".json": a Piper voice, loadable by `piper -m`
SUMMARY = "summary.json"
METADATA = "metadata.csv"
REFERENCE, CLONE = "reference.wav", "voice.json"  # a cloned voice: the clip and what Qwen3-TTS needs to read it
SCORES = (*METRICS, "speaker_similarity")
QUALITY: dict[str, dict[str, Any]] = {
    "medium": {},
    "high": {"model.resblock": "1", "model.resblock_kernel_sizes": "(3, 7, 11)",
             "model.resblock_dilation_sizes": "((1, 3, 5), (1, 3, 5), (1, 3, 5))", "model.upsample_rates": "(8, 8, 2, 2)",
             "model.upsample_initial_channel": 512, "model.upsample_kernel_sizes": "(16, 16, 4, 4)"},
}


def latest(run: Path) -> Path | None:
    """The resume snapshot of the newest Lightning run: every resumed fit opens a `version_<n>` of its own."""
    found = [(int(path.parent.parent.name.split("_")[-1]), path)
             for path in run.glob("lightning_logs/version_*/checkpoints/last.ckpt")]
    return max(found)[1] if found else None


def lightning_device(device: str) -> dict[str, Any]:
    if device.startswith("cuda"):
        return {"trainer.accelerator": "gpu", "trainer.devices": f"[{device.split(':')[1]}]" if ":" in device else 1}
    return {"trainer.accelerator": device, "trainer.devices": 1}


class VoiceDataStage(Stage[VoiceConfig]):
    """Rows of `audio` and `text` from every source, each file found and checked, and the validation share held out."""
    name: ClassVar[str] = "data"

    def identity(self) -> Any:
        # ponytail: the audio counts through each source's directory (sizes, content of files under 1 MiB); a large
        # set rehashes it on every run, so switch to a manifest of hashes when that wait becomes noticeable
        data = self.config.data
        return [self.config.seed, keyed_identity(data.sources), data.validation, audio_file.signatures(data.sources)]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        sources = self.config.data.sources
        rows: list[dict[str, str]] = []
        for index, source in enumerate(sources):
            self.progress.update(index, len(sources), source.name)
            for number, row, path in audio_file.recordings(source):
                text = " ".join(str(row.get("text") or "").split())
                if not text or path is None:
                    raise ValueError(f"Source {index} ({source.name}) row {number}: needs `text` and an `audio` file "
                                     f"that exists (got {row.get('audio')!r} under {audio_file.audio_root(source)})")
                rows.append({"audio": str(path), "text": text})
        if len(rows) < 2:
            raise ValueError(f"{len(rows)} recordings cannot be split into training and validation")
        held_out = held_out_positions(len(rows), self.config.seed, self.config.data.validation)
        split = {"train": [row for i, row in enumerate(rows) if i not in held_out],
                 "validation": [row for i, row in enumerate(rows) if i in held_out]}
        for name, fold in split.items():
            write_jsonl(workdir / f"{name}.jsonl", fold)
        return {**{name: str(workdir / f"{name}.jsonl") for name in split}, "rows": {name: len(fold) for name, fold in split.items()}}


class FitStage(TrainStage[PiperConfig]):
    """`piper.train fit` from the starting weights, then the last checkpoint exported as a voice. A rerun after a
    crash continues from Piper's last epoch snapshot; a finished stage returns its checkpoint directory."""
    requires: ClassVar[tuple[str, ...]] = ("data",)
    sections: ClassVar[tuple[str, ...]] = ("seed", "model", "training", "device")

    def identity(self) -> Any:
        return [super().identity(), self.config.model.init.identity(), piper_process.version()]

    def train(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        checkpoint = workdir / "checkpoint"
        if not (checkpoint / VOICE).is_file():
            model, training = self.config.model, self.config.training
            rows = read_jsonl(Path(inputs["data"]["train"]))
            (workdir / METADATA).write_text("".join(f"{row['audio']}|{row['text']}\n" for row in rows))
            self.progress.update(0, None, "loading the starting checkpoint")
            start = model.init.local()
            # Piper's own held-out split and validation are off: the validate stage measures on this recipe's held-out
            # rows, and a Piper validation run without the UTMOS score fails on its default checkpoint callback.
            # idempotent: Piper sets cudnn.deterministic off, so a GPU run is not bit-identical to an earlier one;
            # a crashed run resumes from the last epoch snapshot rather than repeating the work
            values = {"data.voice_name": self.config.name, "data.csv_path": workdir / METADATA, "data.audio_dir": workdir,
                      "data.espeak_voice": model.espeak_voice, "data.cache_dir": workdir / "cache",
                      "data.config_path": workdir / (VOICE + ".json"), "data.batch_size": training.batch_size,
                      "data.validation_split": 0, "data.num_test_examples": 0, "trainer.limit_val_batches": 0,
                      "model.sample_rate": model.sample_rate, "model.warmstart_ckpt": start, "model.mos_metric": "none",
                      "trainer.max_epochs": training.epochs, "trainer.default_root_dir": workdir / "run",
                      "seed_everything": self.config.seed, **lightning_device(resolve_device(self.config.device)),
                      **QUALITY[model.quality], **training.args}
            # ponytail: a cancelled API job stops only when the fit ends, since Piper reports no progress to this
            # process; read fit.log for the epochs
            self.progress.update(0, None, f"piper.train fit, {training.epochs} epochs (log: {workdir / 'fit.log'})")
            piper_process.fit(values, latest(workdir / "run"), workdir / "fit.log")
            last = latest(workdir / "run")
            if last is None:
                raise piper_process.PiperFailed(f"piper.train fit wrote no checkpoint under {workdir / 'run'}")
            self.progress.update(0, None, "exporting the voice")
            with staged_directory(checkpoint) as staging:
                shutil.copyfile(last, staging / CHECKPOINT_FILE)
                shutil.copyfile(workdir / (VOICE + ".json"), staging / (VOICE + ".json"))
                piper_process.export(staging / CHECKPOINT_FILE, staging / VOICE, workdir / "fit.log")
                write_json(staging / SUMMARY, {"rows": len(rows), "epochs": training.epochs, "init": model.init.identity()})
        return {"checkpoint": str(checkpoint), **read_json(checkpoint / SUMMARY)}


class CloneStage(TrainStage[QwenConfig]):
    """Nothing to train: Qwen3-TTS reads any voice from one recording and its text. The stage picks that recording,
    the training clip UTMOS rates most natural within the configured lengths, since the clone copies the clip's
    sound along with the voice; a filtered clip clones worse, so the clip is kept as recorded."""
    requires: ClassVar[tuple[str, ...]] = ("data",)
    sections: ClassVar[tuple[str, ...]] = ("seed", "model", "reference")

    def train(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        checkpoint = workdir / "checkpoint"
        if not (checkpoint / CLONE).is_file():
            bounds = self.config.reference
            rows = [row for row in read_jsonl(Path(inputs["data"]["train"]))
                    if bounds.min_seconds <= audio_file.seconds(Path(row["audio"])) <= bounds.max_seconds]
            if not rows:
                raise ValueError(f"No training recording lasts {bounds.min_seconds} to {bounds.max_seconds} s; "
                                 "change `reference` or add longer recordings")
            rater = Rater(workdir / "models")
            scored = []
            for index, row in enumerate(rows):
                self.progress.update(index, len(rows), "rating the candidate references")
                scored.append({**row, "utmos": rater(Path(row["audio"]))["utmos"]})
            best = max(scored, key=lambda row: row["utmos"])  # max keeps the first of a tie: the rows' own order
            model = self.config.model
            with staged_directory(checkpoint) as staging:
                shutil.copyfile(best["audio"], staging / REFERENCE)
                write_json(staging / CLONE, {"model": model.name, "revision": model.revision,
                                             "reference_text": best["text"], "sample_rate": qwen_process.RATE})
                write_json(staging / SUMMARY, {"reference": best["audio"], "utmos": best["utmos"],
                                               "seconds": audio_file.seconds(Path(best["audio"])), "candidates": len(rows)})
        return {"checkpoint": str(checkpoint), **read_json(checkpoint / SUMMARY)}


class ListenStage(ValidateStage[VoiceConfig]):
    """The voice reads the held-out texts; Whisper transcribes them and the original recordings. Metrics:
    `wer` (the voice), `reference_wer` (the recordings, the floor `wer` can reach), `duration_ratio` (voice length
    over recording length), and with `validation.phrases`, `phrase_wer`: the worst-heard of the agent's fixed
    replies, which are kept in phrases/ with phrases.json for the agent to play. How each voice sounds, without
    a reference: `utmos` (naturalness) and `dnsmos_sig`, `dnsmos_bak`, `dnsmos_ovrl` (speech distortion, background
    noise, overall; DNSMOS P.835), each from 1 to 5; whose voice it is: `speaker_similarity` to the mean of the
    recordings (0 to 1). `reference_` holds the recordings' own scores and `phrase_` the replies kept."""
    version: ClassVar[int] = 3
    requires: ClassVar[tuple[str, ...]] = ("train", "data")
    sections: ClassVar[tuple[str, ...]] = ("validation", "device")

    def speak(self, checkpoint: Path, jobs: list[tuple[str, Path, int]], workdir: Path) -> None:
        """Each (text, wav, take) read by the voice in `checkpoint` into wav; wavs that exist are kept."""
        raise NotImplementedError(f"{type(self).__name__} must define speak")

    def measure(self, workdir: Path, inputs: dict[str, Outputs]) -> dict[str, float]:
        checkpoint = Path(inputs["train"]["checkpoint"])
        rows = read_jsonl(Path(inputs["data"]["validation"]))[: self.config.validation.rows]
        recordings = [Path(row["audio"]) for name in ("train", "validation") for row in read_jsonl(Path(inputs["data"][name]))]
        self.progress.update(0, len(rows), "speaking the held-out texts")
        self.speak(checkpoint, [(row["text"], workdir / "audio" / f"{index:05d}.wav", 0) for index, row in enumerate(rows)],
                   workdir)
        items: list[dict[str, Any]] = []
        rater, speaker = Rater(workdir / "models"), Speaker(recordings)

        def score(path: Path) -> dict[str, float]:
            return {**rater(path), "speaker_similarity": speaker(path)}

        with Listener(self.config.validation.asr, self.config.device) as listener:
            for index, row in enumerate(rows):
                self.progress.update(index, len(rows), "listening")
                spoken, recorded = workdir / "audio" / f"{index:05d}.wav", Path(row["audio"])
                heard, reference = listener.transcribe(spoken), listener.transcribe(recorded)
                items.append({"text": row["text"], "heard": heard, "wer": wer(row["text"], heard),
                              "reference_heard": reference, "reference_wer": wer(row["text"], reference),
                              "duration_ratio": audio_file.seconds(spoken) / max(audio_file.seconds(recorded), 1e-6),
                              **score(spoken), **{f"reference_{name}": value for name, value in score(recorded).items()}})
            replies = self.phrases(checkpoint, workdir, listener, score)
        self.progress.update(len(rows), len(rows))
        write_jsonl(workdir / "items.jsonl", items)
        averaged = ("wer", "reference_wer", "duration_ratio", *SCORES, *(f"reference_{name}" for name in SCORES))
        metrics = {metric: statistics.fmean(item[metric] for item in items) for metric in averaged}
        if replies:
            metrics["phrase_wer"] = max(reply["wer"] for reply in replies.values())
            metrics |= {f"phrase_{name}": statistics.fmean(reply[name] for reply in replies.values()) for name in SCORES}
        return metrics

    def phrases(self, checkpoint: Path, workdir: Path, listener: Listener,
                score: Callable[[Path], dict[str, float]]) -> dict[str, dict[str, Any]]:
        """Each fixed reply spoken `phrase_takes` times, and each take also put through `phrase_filter` when there is
        one; the take Whisper hears best (then a filtered one, then the first) becomes phrases/<name>.wav. A rerun
        keeps the takes it already has, so the agent plays the file that passed."""
        settings = self.config.validation
        takes_dir = workdir / "phrases" / "takes"
        self.progress.update(0, None, "speaking the replies")
        self.speak(checkpoint, [(text, takes_dir / f"{name}-{take}.wav", take)
                                for name, text in sorted(settings.phrases.items()) for take in range(settings.phrase_takes)],
                   workdir)
        replies: dict[str, dict[str, Any]] = {}
        for name, text in sorted(settings.phrases.items()):
            self.progress.update(0, None, f"listening to the reply {name!r}")
            takes: list[dict[str, Any]] = []
            for take in range(settings.phrase_takes):
                wav = takes_dir / f"{name}-{take}.wav"
                versions = [wav]
                if settings.phrase_filter:
                    # the raw take stays a candidate: a filter that removes hiss can blur the consonants of a name
                    versions.append(wav.with_suffix(".filtered.wav"))
                    if not versions[1].is_file():
                        with atomic_path(versions[1]) as temporary:
                            audio_file.filtered(wav, temporary, settings.phrase_filter)
                for version in versions:
                    heard = listener.transcribe(version)
                    takes.append({"take": take, "filtered": version != wav, "file": version.name, "heard": heard,
                                  "wer": wer(text, heard), **score(version)})
            best = min(takes, key=lambda entry: (entry["wer"], not entry["filtered"], entry["take"]))
            with atomic_path(workdir / "phrases" / f"{name}.wav") as temporary:
                shutil.copyfile(takes_dir / best["file"], temporary)
            replies[name] = {"text": text, "audio": f"phrases/{name}.wav", **best, "takes": takes}
        if replies:
            write_json(workdir / "phrases.json", replies)
        return replies


class PiperListenStage(ListenStage):
    def identity(self) -> Any:
        return [super().identity(), piper_process.version()]

    def speak(self, checkpoint: Path, jobs: list[tuple[str, Path, int]], workdir: Path) -> None:
        for text, wav, _ in jobs:
            if not wav.is_file():
                piper_process.speak(checkpoint / VOICE, text, wav, workdir / "speak.log", {})


class QwenListenStage(ListenStage):
    def identity(self) -> Any:
        return [super().identity(), self.config.seed, qwen_process.version()]

    def speak(self, checkpoint: Path, jobs: list[tuple[str, Path, int]], workdir: Path) -> None:
        qwen_process.speak(read_json(checkpoint / CLONE), checkpoint / REFERENCE,
                   [(text, wav, qwen_process.seed(self.config.seed, text, take)) for text, wav, take in jobs], workdir / "speak.log")
