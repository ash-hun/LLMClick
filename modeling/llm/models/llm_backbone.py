"""What every LLM backbone offers the training methods: chat rendering, token ids, logits and generation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from modeling.llm.models.backbone_config import LLMBackboneConfig
from modeling.tuning.tuning_backbone import Backbone
from core.device_selector import Device
from core.file_store import write_json

if TYPE_CHECKING:
    from modeling.llm.models.heads.decision_head import DecisionHead

Message = dict[str, str]
HEAD_WEIGHTS, HEAD_CONFIG = "head.safetensors", "head.json"
MISSING_BITSANDBYTES = "training.adapter.quantize needs bitsandbytes: `uv sync --extra quantize`"
GGUF_FILE = "model.gguf"  # the file llm_quantize's gguf scheme writes next to the tokenizer


def prequantized(origin: Path) -> dict[str, Any] | None:
    """Loader arguments for a local model directory holding quantized weights (bitsandbytes, compressed-tensors or
    GGUF), or None for ordinary weights."""
    if (origin / GGUF_FILE).is_file():
        return {"gguf_file": GGUF_FILE}  # dequantized on load: transformers reads GGUF but cannot run it quantized
    config = origin / "config.json"
    saved = json.loads(config.read_text()).get("quantization_config") if config.is_file() else None
    if saved is None:
        return None
    if saved.get("quant_method") == "compressed-tensors":
        from transformers import CompressedTensorsConfig
        # decompressed into dense layers on load: the packed kernels exist for vLLM on CUDA, and measuring the
        # dequantized weights scores the quantization error itself, the same on every device
        return {"quantization_config": CompressedTensorsConfig(run_compressed=False)}
    return {}


def quantization_config(scheme: str) -> Any:
    """bitsandbytes settings of a QLoRA base: NF4 with double quantization and bf16 compute, or LLM.int8()."""
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        raise ImportError(MISSING_BITSANDBYTES) from None
    from transformers import BitsAndBytesConfig
    if scheme == "nf4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,  # type: ignore[no-untyped-call]
                                  bnb_4bit_compute_dtype=torch.bfloat16)
    return BitsAndBytesConfig(load_in_8bit=True)  # type: ignore[no-untyped-call]


@dataclass(frozen=True)
class Markers:
    """Single tokens that give a decision prompt its structure. An architecture names tokens its tokenizer already
    has and ordinary text hardly uses; plain words in their place score worse (Jeeves)."""
    state: str
    question: str
    option: str
    option_end: str
    decide: str
    think: str
    think_end: str


QWEN_MARKERS = Markers("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>", "<think>", "</think>")


class LLMBackbone(Backbone):
    quantizable: ClassVar[bool] = True
    loader: ClassVar[Any]                        # the transformers Auto class that builds this architecture
    frozen: ClassVar[tuple[str, ...]] = ()       # parameter-name fragments that text training must not update
    markers: ClassVar[Markers | None] = None     # needed by decision recipes; set by architectures that support them
    config: LLMBackboneConfig
    head: "DecisionHead | None" = None           # set when `model.head` names one

    def __init__(self, config: LLMBackboneConfig) -> None:
        super().__init__(config)
        # Token ids by text: the length check before training, every epoch's loss and the evaluation all tokenize
        # the same rows, and tokenizing once is enough. The cache lives as long as the loaded tokenizer; it holds
        # one entry per distinct text, so its size is that of the tokenized data.
        self._tokens: dict[tuple[str, ...], list[int]] = {}

    def load(self, device: Device, checkpoint: Path | None = None) -> None:
        origin, revision = self.origin(checkpoint)
        self._tokens.clear()
        self.tokenizer = AutoTokenizer.from_pretrained(origin, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.quantize is not None and checkpoint is None:
            self.model = self.quantized(origin, revision, device)
        elif (quantized := prequantized(Path(origin))) is not None:
            # weights `llm_quantize` wrote: they keep their own dtypes, and a quantized model is placed, never moved
            self.model = self.loader.from_pretrained(origin, revision=revision, device_map={"": device}, **quantized)
        else:
            self.model = self.loader.from_pretrained(origin, revision=revision, dtype=self.dtype).to(device)
        for name, parameter in self.model.named_parameters():
            if any(fragment in name for fragment in self.frozen):
                parameter.requires_grad_(False)
        self.device = device
        if self.config.head is not None:
            self.load_head(Path(origin))

    def quantized(self, origin: str, revision: str | None, device: Device) -> Any:
        """The base weights in 4 or 8 bits for QLoRA, everything else in `dtype`. bitsandbytes kernels exist for CUDA
        only, so another device stops here rather than in the first forward pass."""
        if not device.startswith("cuda"):
            raise ValueError(f"training.adapter.quantize needs a CUDA device, not {device!r}")
        from peft import prepare_model_for_kbit_training
        model = self.loader.from_pretrained(origin, revision=revision, dtype=self.dtype, device_map={"": device},
                                            quantization_config=quantization_config(str(self.quantize)))
        return prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)  # type: ignore[no-untyped-call]

    def exported(self) -> Any:
        """A QLoRA adapter is folded into the base reloaded in `dtype` on the CPU, not into the 4-bit weights: merging
        into quantized weights rounds the update away, and the checkpoint must load like any other model."""
        if self.quantize is None or not hasattr(self.model, "merge_and_unload"):
            return super().exported()
        from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
        # ponytail: the full base is held in CPU memory at `dtype` while merging (about 4 bytes a weight); stream
        # the merge layer by layer if a model ever outgrows the host's memory
        origin, revision = self.origin(None)
        base = self.loader.from_pretrained(origin, revision=revision, dtype=self.dtype)
        merged = get_peft_model(base, self.model.peft_config["default"])
        set_peft_model_state_dict(merged, get_peft_model_state_dict(self.model))
        return merged.merge_and_unload()

    def load_head(self, origin: Path) -> None:
        """A fresh head on a base model; the saved head (weights, option codes, temperature) on a checkpoint."""
        from modeling.llm.models.heads import HEADS
        saved = json.loads((origin / HEAD_CONFIG).read_text()) if (origin / HEAD_CONFIG).exists() else None
        if saved is not None and saved["head"] != self.config.head:
            raise ValueError(f"{origin} was trained with head {saved['head']!r}, the config asks for {self.config.head!r}")
        self.head = HEADS.get(str(self.config.head))(self, saved)
        assert self.head is not None
        if saved is not None:
            self.head.load_state_dict(load_file(str(origin / HEAD_WEIGHTS)))
        self.head.to(self.device)

    def modules(self) -> dict[str, torch.nn.Module]:
        return {**super().modules(), **({"head": self.head} if self.head is not None else {})}

    def save(self, path: Path) -> None:
        self.exported().save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        if self.head is not None:
            save_file({name: value.detach().cpu().contiguous() for name, value in self.head.state_dict().items()},
                      str(path / HEAD_WEIGHTS))
            write_json(path / HEAD_CONFIG, {"head": self.config.head, **self.head.settings()})

    def release(self) -> None:
        self.head = None
        self._tokens.clear()
        super().release()

    def cached(self, key: tuple[str, ...], text: str) -> list[int]:
        """Token ids of `text` (no special tokens), tokenized once per distinct text; a copy, so callers may extend it."""
        ids = self._tokens.get(key)
        if ids is None:
            ids = self._tokens[key] = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return list(ids)

    def token_ids(self, text: str) -> list[int]:
        return self.cached(("text", text), text)

    def decision_markers(self) -> Markers:
        if self.markers is None:
            raise ValueError(f"architecture {self.config.architecture!r} defines no decision markers; decision recipes "
                             "need the backbone class to set `markers`")
        return self.markers

    def neutral(self, text: str) -> str:
        """User text with every control token of this tokenizer broken up, so it cannot forge a marker or a turn."""
        for token in sorted({*self.tokenizer.all_special_tokens, *self.tokenizer.get_added_vocab()}, key=len, reverse=True):
            if token in text:
                text = text.replace(token, f"{token[:1]}\u200b{token[1:]}")
        return text

    def think_prompt(self, content: str) -> list[int]:
        """One user turn followed by an open reasoning block. Some templates open the block themselves (Qwen3.5);
        others leave that to the model (Qwen3), so it is added here."""
        think = self.decision_markers().think
        text: str = self.tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                       add_generation_prompt=True, **{**self.config.template, "enable_thinking": True})
        return self.token_ids(text if text.endswith(f"{think}\n") else f"{text}{think}\n")

    def hidden(self, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Final hidden states of right-padded sequences, with their token ids and attention mask. The language-model
        head is not applied: over a whole batch its output (tokens x vocabulary) is by far the largest tensor."""
        ids, mask = self.padded(sequences)
        inner = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model  # under LoRA
        states: torch.Tensor = inner.base_model(input_ids=ids, attention_mask=mask).last_hidden_state
        return states, ids, mask

    def next_token_log_probabilities(self, hidden: torch.Tensor, ids: torch.Tensor, graded: torch.Tensor,
                                     temperature: float = 1.0) -> torch.Tensor:
        """log p(token | everything before it) for the tokens `graded` marks, zero elsewhere; shape (rows, tokens - 1).
        The language-model head runs only on the marked positions."""
        selected = graded[:, 1:].bool()
        out = hidden.new_zeros(selected.shape, dtype=torch.float32)
        if bool(selected.any()):
            logits = self.model.get_output_embeddings()(hidden[:, :-1][selected]).float() / temperature
            out[selected] = torch.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:][selected].unsqueeze(-1)).squeeze(-1)
        return out

    def render(self, messages: list[Message], generation_prompt: bool) -> list[int]:
        """Token ids of a conversation in the model's own chat format."""
        key = ("chat", str(generation_prompt), json.dumps(messages, ensure_ascii=False, sort_keys=True))
        ids = self._tokens.get(key)
        if ids is None:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt,
                                                      **self.config.template)
            ids = self._tokens[key] = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return list(ids)

    def padded(self, sequences: list[list[int]], left: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(sequence) for sequence in sequences)
        ids = torch.full((len(sequences), width), int(self.tokenizer.pad_token_id), dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            span = slice(width - len(sequence), width) if left else slice(0, len(sequence))
            ids[row, span] = torch.tensor(sequence, dtype=torch.long)
            mask[row, span] = 1
        return ids.to(self.device), mask.to(self.device)

    @torch.no_grad()
    def generate(self, prompts: list[list[int]], max_new_tokens: int, samples: int = 1, temperature: float = 0.0,
                 stop: int | None = None, suppress: list[int] | None = None) -> list[list[list[int]]]:
        """Per prompt, `samples` completions as token ids cut at the end token (or at `stop`); temperature 0 decodes
        greedily. `suppress` lists tokens that may never be produced."""
        ids, mask = self.padded(prompts, left=True)
        sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
        training = self.model.training
        self.model.eval()
        output = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                     num_return_sequences=samples, pad_token_id=self.tokenizer.pad_token_id,
                                     eos_token_id=[self.tokenizer.eos_token_id, *([stop] if stop is not None else [])],
                                     suppress_tokens=suppress, **sampling)
        self.model.train(training)
        completions = output[:, ids.shape[1]:].tolist()
        ends = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, stop}
        cut = [next((tokens[:i] for i, token in enumerate(tokens) if token in ends), tokens) for tokens in completions]
        return [cut[start:start + samples] for start in range(0, len(cut), samples)]

    def text(self, tokens: list[int]) -> str:
        decoded: str = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return decoded
