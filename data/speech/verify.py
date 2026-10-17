"""Stage `verify`: length, clipping and silence of every recording, then Whisper's transcript against its text."""

from pathlib import Path
from contextlib import nullcontext
from typing import Any, ClassVar

from data.rows import Row, cached, keep, read_rows, trail, write_rows
from data.speech.config import SpeechConfig, SpeechVerifySettings
from modeling.tts.asr import Listener, wer
from core.stage import Outputs, Stage
from modeling.tts import audio

VERIFIED = "verified.jsonl"
REJECTED = "rejected.jsonl"


def failure(measured: dict[str, float], settings: SpeechVerifySettings) -> str | None:
    """Pure: why a recording with these measurements is rejected, or None; cheapest checks first."""
    if measured["seconds"] < settings.min_seconds:
        return "too short"
    if measured["seconds"] > settings.max_seconds:
        return "too long"
    if measured["clipped"] > settings.max_clipped:
        return "clipped"
    if measured["silent"] > settings.max_silent:
        return "silent"
    if "wer" in measured and measured["wer"] > settings.max_wer:
        return "misheard"
    return None


class SpeechVerifyStage(Stage[SpeechConfig]):
    name: ClassVar[str] = "verify"
    requires: ClassVar[tuple[str, ...]] = ("synthesize",)
    sections: ClassVar[tuple[str, ...]] = ("verify",)

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        settings = self.config.verify
        rows = read_rows(Path(inputs["synthesize"]["rows"]))
        verified: list[Row] = []
        rejected: list[Row] = []
        reasons: dict[str, int] = {}
        with Listener(settings.asr, self.config.device) if settings.asr else nullcontext() as listener:
            for done, row in enumerate(rows):
                self.progress.update(done, len(rows), "listening")
                measured: dict[str, Any] = audio.measure(*audio.read(Path(row["audio"])))
                if listener is not None and failure(measured, settings) is None:
                    entry = cached(workdir / "cache", row["id"])  # Whisper is the slow part; a rerun keeps its transcripts
                    if entry is None:
                        entry = keep(workdir / "cache", row["id"], {"heard": listener.transcribe(Path(row["audio"]))})
                    measured.update(heard=entry["heard"], wer=wer(row["text"], entry["heard"]))
                reason = failure(measured, settings)
                if reason is None:
                    verified.append({**row, **measured})
                else:
                    reasons[reason] = reasons.get(reason, 0) + 1
                    rejected.append({**row, **measured, "rejected": reason})
        self.progress.update(len(rows), len(rows))
        write_rows(workdir / VERIFIED, verified)
        write_rows(workdir / REJECTED, rejected)
        summary = {"count": len(verified), "rejected": len(rejected), "reasons": reasons}
        return {"rows": str(workdir / VERIFIED), "rejected_rows": str(workdir / REJECTED), **summary,
                "trail": trail(inputs, self.name, summary)}
