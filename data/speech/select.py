"""Stage `select`: one recording per text, written as a Piper training set (wavs/, metadata.csv, rows.jsonl) with its manifest."""

import os
import shutil
from pathlib import Path
from typing import ClassVar

from data.rows import Row, normalized, read_rows, trail, write_rows
from data.speech.config import SpeechConfig
from data.teachers import teacher_from
from core.utils.files import write_json
from core.stage import Outputs, Stage

ROWS = "rows.jsonl"
METADATA = "metadata.csv"
MANIFEST = "manifest.json"
WAVS = "wavs"


def chosen(rows: list[Row], per_text: int | None = 1) -> list[Row]:
    """Pure: per text the `per_text` recordings Whisper understood best (then the first sample), in a stable order."""
    best: dict[str, list[Row]] = {}
    for row in sorted(rows, key=lambda row: (normalized(row["text"]), float(row.get("wer", 0)), int(row["sample"]), row["id"])):
        kept = best.setdefault(normalized(row["text"]), [])
        if per_text is None or len(kept) < per_text:
            kept.append(row)
    return sorted((row for kept in best.values() for row in kept), key=lambda row: row["id"])


def placed(source: Path, target: Path) -> None:
    """`target` holds the recording: a hard link where the file system allows one, else a copy; never half a file."""
    if target.is_file():
        return
    temporary = target.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    temporary.replace(target)


class SpeechSelectStage(Stage[SpeechConfig]):
    name: ClassVar[str] = "select"
    requires: ClassVar[tuple[str, ...]] = ("verify",)
    sections: ClassVar[tuple[str, ...]] = ("select", "licence")

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        config = self.config
        verified = read_rows(Path(inputs["verify"]["rows"]))
        unique = chosen(verified, config.select.per_text)
        rows = unique[: config.select.max_rows]
        (workdir / WAVS).mkdir(exist_ok=True)
        for done, row in enumerate(rows):
            self.progress.update(done, len(rows), "writing the dataset")
            placed(Path(row["audio"]), workdir / WAVS / f"{row['id']}.wav")
        # paths relative to the dataset directory, so it can be moved, and Piper reads metadata.csv as it is
        (workdir / METADATA).write_text("".join(f"{WAVS}/{row['id']}.wav|{row['text']}\n" for row in rows))
        write_rows(workdir / ROWS, [{"audio": f"{WAVS}/{row['id']}.wav", "text": row["text"]} for row in rows])
        counts = {"verified": len(verified), "duplicate_text": len(verified) - len(unique), "over_max_rows": len(unique) - len(rows),
                  "rows": len(rows), "seconds": round(sum(float(row["seconds"]) for row in rows), 3)}
        stages = trail(inputs, self.name, {"counts": counts})
        manifest = {"recipe": config.recipe, "name": config.name, "licence": config.licence, "counts": counts, "stages": stages,
                    "voice": config.voice.model_dump(mode="json"),
                    "teacher": teacher_from(config.teacher).identity() if config.teacher else None,
                    "seeds": [seed.model_dump(mode="json") for seed in config.seeds],
                    "settings": {"prompts": config.prompts.model_dump(mode="json"), "verify": config.verify.model_dump(mode="json"),
                                 "select": config.select.model_dump(mode="json")}}
        write_json(workdir / MANIFEST, manifest)
        self.progress.update(len(rows), len(rows))
        return {"rows": str(workdir / ROWS), "metadata": str(workdir / METADATA), "manifest": str(workdir / MANIFEST),
                "count": len(rows), "counts": counts, "trail": stages}
