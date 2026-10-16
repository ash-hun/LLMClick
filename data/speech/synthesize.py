"""Stage `synthesize`: every prompt read aloud by the voice, each recording kept under its natural key."""

from pathlib import Path
from typing import Any, ClassVar

from data.rows import Row, key, read_rows, trail, write_rows
from data.speech.config import SpeechConfig
from core.stage import Outputs, Stage
from modeling.tts import piper

SPOKEN = "spoken.jsonl"


def spoken_text(text: str) -> str:
    """One line without the metadata.csv delimiter: what is read is what the dataset says was read."""
    return " ".join(text.replace("|", ",").split())


class SynthesizeStage(Stage[SpeechConfig]):
    """The voice and its settings are in the fingerprint, so a file name only needs the prompt and the sample; a
    rerun after a crash reads only the texts that have no file yet."""
    name: ClassVar[str] = "synthesize"
    requires: ClassVar[tuple[str, ...]] = ("prompts",)
    sections: ClassVar[tuple[str, ...]] = ("voice",)

    def identity(self) -> Any:
        return [super().identity(), self.config.voice.model.identity(), piper.version()]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        settings = self.config.voice
        self.progress.update(0, None, "loading the voice")
        voice = settings.model.local(".json")
        prompts = read_rows(Path(inputs["prompts"]["rows"]))
        jobs = [(prompt, sample) for prompt in prompts for sample in range(settings.samples)]
        rows: list[Row] = []
        made = 0
        for done, (prompt, sample) in enumerate(jobs):
            self.progress.update(done, len(jobs), "speaking")
            identifier, text = key("speech", prompt["id"], sample), spoken_text(prompt["instruction"])
            wav = workdir / "audio" / f"{identifier}.wav"
            if not text:
                continue
            if not wav.is_file():
                piper.speak(voice, text, wav, workdir / "speak.log", settings.scales())
                made += 1
            rows.append({"id": identifier, "prompt_id": prompt["id"], "seed_id": prompt.get("seed_id"), "sample": sample,
                         "text": text, "audio": str(wav)})
        self.progress.update(len(jobs), len(jobs))
        write_rows(workdir / SPOKEN, rows)
        summary = {"count": len(rows), "prompts": len(prompts), "spoken_now": made}
        return {"rows": str(workdir / SPOKEN), **summary, "trail": trail(inputs, self.name, summary)}
