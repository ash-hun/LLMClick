"""Adding words to a byte-level BPE tokenizer as merges, so a frequent word costs one token; also measuring the gain."""

import json
from typing import Any
from collections import Counter


def backend_state(tokenizer: Any) -> dict[str, Any]:
    state: dict[str, Any] = json.loads(tokenizer.backend_tokenizer.to_str())
    model = state.get("model") or {}
    if model.get("type") != "BPE" or "ByteLevel" not in json.dumps(state.get("pre_tokenizer")):
        raise ValueError("model.tokenizer.tokens needs a byte-level BPE tokenizer (Qwen, Llama 3, GPT-2 and the like)")
    return state


def pieces(tokenizer: Any, text: str) -> list[str]:
    """`text` as the pre-tokenizer splits it, in byte-level form (a leading space is part of the piece that follows it)."""
    return [piece for piece, _ in tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)]


def extend(tokenizer: Any, words: list[str]) -> dict[str, list[int]]:
    """Make each word one token, at the start of a text and after a space alike, by appending merges after the
    tokenizer's own: its merges still apply first, so every other text tokenizes as before, and the new ones join
    what is left of the word. Returns each new vocabulary entry with the ids of the pieces it joins under the
    tokenizer as it was, for initialising its embedding. Words already single tokens add nothing, so applying the
    same words again changes nothing. A word that is more than one pre-token (with punctuation inside) is skipped."""
    state = backend_state(tokenizer)
    model = state["model"]
    vocab: dict[str, int] = model["vocab"]
    # transformers keeps the special tokens beside the BPE vocabulary; listing them in it as well stops the library
    # from renumbering them when the vocabulary grows past them
    vocab.update({token["content"]: token["id"] for token in state["added_tokens"]})
    merges: list[Any] = model["merges"]
    known = {(pair[0], pair[1]) if isinstance(pair, list) else tuple(pair.split(" ", 1)) for pair in merges}
    original = tokenizer.backend_tokenizer.model
    next_id = max(vocab.values()) + 1
    created: dict[str, list[int]] = {}
    for word in dict.fromkeys(word.strip() for word in words if word.strip()):
        for variant in (word, f" {word}"):
            split = pieces(tokenizer, variant)
            if len(split) != 1:
                continue
            parts = [token.value for token in original.tokenize(split[0])]
            current = parts[0]
            for index, part in enumerate(parts[1:], start=1):
                joined = current + part
                if (current, part) not in known:
                    merges.append([current, part] if not merges or isinstance(merges[0], list) else f"{current} {part}")
                    known.add((current, part))
                if joined not in vocab:
                    vocab[joined] = next_id
                    created[joined] = [vocab[piece] for piece in parts[: index + 1]]
                    next_id += 1
                current = joined
    if created:
        from tokenizers import Tokenizer
        tokenizer._tokenizer = Tokenizer.from_str(json.dumps(state))
    return created


def chars_per_token(tokenizer: Any, texts: list[str]) -> float:
    """Characters of text per token: higher means a text fits in fewer tokens."""
    tokens = sum(len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts)
    return sum(len(text) for text in texts) / max(tokens, 1)


def candidates(tokenizer: Any, texts: list[str], count: int, min_frequency: int) -> list[tuple[str, int, int]]:
    """The words whose joining saves the most tokens on `texts`: (word, occurrences, tokens saved), most saved first.
    A word is a pre-token without its leading space; one the tokenizer already keeps whole saves nothing."""
    counts: Counter[str] = Counter(piece for text in texts for piece in pieces(tokenizer, text))
    words: Counter[str] = Counter()
    lengths: dict[str, int] = {}
    model = tokenizer.backend_tokenizer.model
    for piece, frequency in counts.items():
        word = tokenizer.convert_tokens_to_string([piece]).strip()
        if not any(character.isalpha() for character in word):
            continue
        size = len(model.tokenize(piece))
        if size > 1:
            words[word] += frequency
            lengths[word] = max(lengths.get(word, 0), size)
    ranked = sorted(((word, frequency, frequency * (lengths[word] - 1)) for word, frequency in words.items()
                     if frequency >= min_frequency), key=lambda item: (-item[2], item[0]))
    return ranked[:count]
