import random

from jeff import voice

PRONUNCIATIONS = {"buy": ["B AY"], "by": ["B AY"], "bye": ["B AY"], "out": ["AW T"], "buyout": ["B AY AW T"],
                  "sell": ["S EH L"], "cell": ["S EH L"], "side": ["S AY D"], "sighed": ["S AY D"], "ng": ["EH N JH IY"]}


def sounds(common: set[str] | None = None) -> voice.SoundAlikes:
    return voice.SoundAlikes(PRONUNCIATIONS, common if common is not None else set(PRONUNCIATIONS))


def test_homophones_and_splits_come_from_the_dictionary() -> None:
    assert sounds().homophones("buy") == ["by", "bye"]
    assert "by out" in sounds().splits("buyout") and "bye out" in sounds().splits("buyout")
    assert sounds(common={"buy", "sell", "side", "out"}).homophones("buy") == []  # rare spellings are not used


def test_garble_changes_words_to_sound_alikes_only() -> None:
    garbled = sounds().garble("sell side", random.Random(1), changes=2)
    assert garbled in {"cell sighed", "cell side", "sell sighed"}
    assert sounds().garble("hello there", random.Random(1), changes=1) is None


def test_phrase_rows_label_the_true_name_among_options() -> None:
    row = {"id": "7", "utt": "open sell side research", "annot_utt": "open [business_name : sell side] research",
           "scenario": "general", "intent": "general_quirky"}
    values = {"business_name": ["sell side", "buy side", "cell towers", "research desk", "side street", "deal room"]}
    made = voice.phrase_row(row, values, sounds(), random.Random(3), "train")
    assert made is not None and "sell side" not in made["state"]
    assert made["question"]["criteria"][made["label"]] == "sell side"
    assert made["question"]["instructions"] == "Which business did the user mean?"


def test_question_intents_are_described_as_questions() -> None:
    assert voice.intent_description("qa_definition").startswith("Not a command: the user is asking a question")
    assert voice.intent_description("iot_hue_lightoff") == "Smart home lights light off"
