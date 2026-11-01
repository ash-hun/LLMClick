"""Stages of `llm_tokenizer`: `train` picks the words and writes the extended tokenizer, `validate` measures the gain."""

import shutil
from pathlib import Path
from typing import Any, ClassVar

from transformers import AutoTokenizer

from modeling.llm.models.vocabulary_extender import candidates, chars_per_token, extend
from core.file_store import directory_signature, read_json, read_jsonl, staged_directory, write_json
from modeling.vocabulary.vocabulary_config import TokenizerConfig
from modeling.modeling_stages import TrainStage, ValidateStage
from core.stage_base import Outputs

TOKENS, CANDIDATES, SUMMARY, CHECKPOINT = "tokens.txt", "candidates.json", "summary.json", "checkpoint"


def base_tokenizer(config: TokenizerConfig) -> Any:
    model = config.model
    local = Path(model.name).is_dir()
    return AutoTokenizer.from_pretrained(model.name, revision=None if local else model.revision)


def texts(path: str) -> list[str]:
    return [str(row["text"]) for row in read_jsonl(Path(path))]


class VocabularyStage(TrainStage[TokenizerConfig]):
    """Count the training rows' words, keep those the tokenizer splits and that save the most tokens, write them to
    `tokens.txt` (what `model.tokenizer.tokens_file` reads) and the tokenizer extended with them to `checkpoint/`."""
    requires: ClassVar[tuple[str, ...]] = ("data",)
    sections: ClassVar[tuple[str, ...]] = ("seed", "recipe", "model", "vocabulary")

    def identity(self) -> Any:
        return [super().identity(), directory_signature(Path(self.config.model.name))]

    def train(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        checkpoint, summary = workdir / CHECKPOINT, workdir / SUMMARY
        if not (summary.exists() and checkpoint.is_dir()):
            shutil.rmtree(checkpoint, ignore_errors=True)  # a crash between writing the tokenizer and the summary
            settings, tokenizer = self.config.vocabulary, base_tokenizer(self.config)
            self.progress.update(0, None, "counting words")
            ranked = candidates(tokenizer, texts(inputs["data"]["train"]), settings.count, settings.min_frequency)
            (workdir / TOKENS).write_text("".join(f"{word}\n" for word, _, _ in ranked), encoding="utf-8")
            write_json(workdir / CANDIDATES, [{"word": word, "occurrences": count, "tokens_saved": saved}
                                              for word, count, saved in ranked])
            self.progress.update(0, None, "extending the tokenizer")
            created = extend(tokenizer, [word for word, _, _ in ranked])
            with staged_directory(checkpoint) as staging:
                tokenizer.save_pretrained(staging)
            write_json(summary, {"words": len(ranked), "vocabulary_added": len(created)})
        return {"checkpoint": str(checkpoint), "tokens": str(workdir / TOKENS), "run": str(workdir), **read_json(summary)}


class VocabularyMeasureStage(ValidateStage[TokenizerConfig]):
    """Characters per token on the held-out rows, with the base and with the extended tokenizer."""
    requires: ClassVar[tuple[str, ...]] = ("train", "data")

    def measure(self, workdir: Path, inputs: dict[str, Outputs]) -> dict[str, float]:
        held_out = texts(inputs["data"]["validation"])
        base = chars_per_token(base_tokenizer(self.config), held_out)
        extended = chars_per_token(AutoTokenizer.from_pretrained(inputs["train"]["checkpoint"]), held_out)
        return {"base_chars_per_token": base, "chars_per_token": extended, "token_reduction": 1 - base / extended,
                "words": float(inputs["train"]["words"])}
