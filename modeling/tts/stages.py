"""The stages of `tts_piper`: recordings -> Piper fine-tuning and export -> a listening test of the exported voice."""

import shutil
import statistics
from pathlib import Path
from typing import Any, ClassVar

from core.utils.files import atomic_path, read_json, read_jsonl, staged_directory, write_json, write_jsonl
from modeling.tts.config import CHECKPOINT_FILE, PiperConfig
from modeling.stages import TrainStage, ValidateStage, held_out_positions
from modeling.tts.quality import METRICS, Rater
from core.utils.device import resolve_device
from modeling.tts.asr import Listener, wer
from modeling.config import keyed_identity
from modeling.tts import audio, piper
from core.stage import Outputs, Stage

VOICE = "voice.onnx"  # with VOICE + ".json": a Piper voice, loadable by `piper -m`
SUMMARY = "summary.json"
METADATA = "metadata.csv"
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


class VoiceDataStage(Stage[PiperConfig]):
    """Rows of `audio` and `text` from every source, each file found and checked, and the validation share held out."""
    name: ClassVar[str] = "data"

    def identity(self) -> Any:
        # ponytail: the audio counts through each source's directory (sizes, content of files under 1 MiB); a large
        # set rehashes it on every run, so switch to a manifest of hashes when that wait becomes noticeable
        data = self.config.data
        return [self.config.seed, keyed_identity(data.sources), data.validation, audio.signatures(data.sources)]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        sources = self.config.data.sources
        rows: list[dict[str, str]] = []
        for index, source in enumerate(sources):
            self.progress.update(index, len(sources), source.name)
            for number, row, path in audio.recordings(source):
                text = " ".join(str(row.get("text") or "").split())
                if not text or path is None:
                    raise ValueError(f"Source {index} ({source.name}) row {number}: needs `text` and an `audio` file "
                                     f"that exists (got {row.get('audio')!r} under {audio.audio_root(source)})")
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
        return [super().identity(), self.config.model.init.identity(), piper.version()]

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
            piper.fit(values, latest(workdir / "run"), workdir / "fit.log")
            last = latest(workdir / "run")
            if last is None:
                raise piper.PiperFailed(f"piper.train fit wrote no checkpoint under {workdir / 'run'}")
            self.progress.update(0, None, "exporting the voice")
            with staged_directory(checkpoint) as staging:
                shutil.copyfile(last, staging / CHECKPOINT_FILE)
                shutil.copyfile(workdir / (VOICE + ".json"), staging / (VOICE + ".json"))
                piper.export(staging / CHECKPOINT_FILE, staging / VOICE, workdir / "fit.log")
                write_json(staging / SUMMARY, {"rows": len(rows), "epochs": training.epochs, "init": model.init.identity()})
        return {"checkpoint": str(checkpoint), **read_json(checkpoint / SUMMARY)}


class ListenStage(ValidateStage[PiperConfig]):
    """The exported voice reads the held-out texts; Whisper transcribes them and the original recordings. Metrics:
    `wer` (the voice), `reference_wer` (the recordings, the floor `wer` can reach), `duration_ratio` (voice length
    over recording length), and with `validation.phrases`, `phrase_wer`: the worst-heard of the agent's fixed
    replies, which are kept in phrases/ with phrases.json for the agent to play. How clean each voice sounds, without
    a reference: `utmos` (naturalness) and `dnsmos_sig`, `dnsmos_bak`, `dnsmos_ovrl` (speech distortion, background
    noise, overall; DNSMOS P.835), each from 1 to 5, with `reference_` for the recordings and `phrase_` for the
    replies kept."""
    version: ClassVar[int] = 2
    requires: ClassVar[tuple[str, ...]] = ("train", "data")
    sections: ClassVar[tuple[str, ...]] = ("validation", "device")

    def identity(self) -> Any:
        return [super().identity(), piper.version()]

    def measure(self, workdir: Path, inputs: dict[str, Outputs]) -> dict[str, float]:
        voice = Path(inputs["train"]["checkpoint"]) / VOICE
        rows = read_jsonl(Path(inputs["data"]["validation"]))[: self.config.validation.rows]
        items: list[dict[str, Any]] = []
        rater = Rater(workdir / "models")
        with Listener(self.config.validation.asr, self.config.device) as listener:
            for index, row in enumerate(rows):
                self.progress.update(index, len(rows), "speaking and listening")
                spoken, recorded = workdir / "audio" / f"{index:05d}.wav", Path(row["audio"])
                if not spoken.is_file():
                    piper.speak(voice, row["text"], spoken, workdir / "speak.log", {})
                heard, reference = listener.transcribe(spoken), listener.transcribe(recorded)
                items.append({"text": row["text"], "heard": heard, "wer": wer(row["text"], heard),
                              "reference_heard": reference, "reference_wer": wer(row["text"], reference),
                              "duration_ratio": audio.seconds(spoken) / max(audio.seconds(recorded), 1e-6),
                              **rater(spoken), **{f"reference_{name}": value for name, value in rater(recorded).items()}})
            replies = self.phrases(voice, workdir, listener, rater)
        self.progress.update(len(rows), len(rows))
        write_jsonl(workdir / "items.jsonl", items)
        averaged = ("wer", "reference_wer", "duration_ratio", *METRICS, *(f"reference_{name}" for name in METRICS))
        metrics = {metric: statistics.fmean(item[metric] for item in items) for metric in averaged}
        if replies:
            metrics["phrase_wer"] = max(reply["wer"] for reply in replies.values())
            metrics |= {f"phrase_{name}": statistics.fmean(reply[name] for reply in replies.values()) for name in METRICS}
        return metrics

    def phrases(self, voice: Path, workdir: Path, listener: Listener, rater: Rater) -> dict[str, dict[str, Any]]:
        """Each fixed reply spoken `phrase_takes` times; the take Whisper hears best (then the first) becomes
        phrases/<name>.wav. A rerun keeps the takes it already has, so the agent plays the file that passed."""
        settings = self.config.validation
        replies: dict[str, dict[str, Any]] = {}
        for name, text in sorted(settings.phrases.items()):
            self.progress.update(0, None, f"speaking the reply {name!r}")
            takes: list[dict[str, Any]] = []
            for take in range(settings.phrase_takes):
                wav = workdir / "phrases" / "takes" / f"{name}-{take}.wav"
                if not wav.is_file():
                    piper.speak(voice, text, wav, workdir / "speak.log", {})
                heard = listener.transcribe(wav)
                takes.append({"take": take, "heard": heard, "wer": wer(text, heard), **rater(wav)})
            best = min(takes, key=lambda entry: (entry["wer"], entry["take"]))
            with atomic_path(workdir / "phrases" / f"{name}.wav") as temporary:
                shutil.copyfile(workdir / "phrases" / "takes" / f"{name}-{best['take']}.wav", temporary)
            replies[name] = {"text": text, "audio": f"phrases/{name}.wav", **best, "takes": takes}
        if replies:
            write_json(workdir / "phrases.json", replies)
        return replies
