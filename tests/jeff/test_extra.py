import pytest

from jeff import extra
from jeff.data import validate


def test_snli_neutral_label() -> None:
    raw = {"premise": "A person on a horse jumps over a broken down airplane.",
           "hypothesis": "A person is training his horse for a competition.", "label": 1}
    row = extra.convert_snli(0, raw)
    assert row is not None
    assert row["state"] == ("Premise: A person on a horse jumps over a broken down airplane.\n"
                             "Hypothesis: A person is training his horse for a competition.")
    assert row["question"]["instructions"] == extra.SNLI_INSTRUCTIONS
    assert list(row["question"]["criteria"]) == ["entailment", "neutral", "contradiction"]
    assert row["label"] == row["target"] == "neutral"
    assert row["id"] == "extra-snli-0" == row["family"]


def test_snli_drops_unlabelled_rows() -> None:
    raw = {"premise": "p", "hypothesis": "h", "label": -1}
    assert extra.convert_snli(5, raw) is None


def test_commonsense_qa() -> None:
    raw = {"id": "075e483d21c29a511267ef62bedc0461",
           "question": ("The sanctions against the school were a punishing blow, and they seemed to what "
                         "the efforts the school had made to change?"),
           "question_concept": "punishing",
           "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["ignore", "enforce", "authoritarian", "yell at", "avoid"]},
           "answerKey": "A"}
    row = extra.convert_commonsense_qa(raw)
    assert row["state"] == raw["question"]
    assert row["question"]["instructions"] == extra.MULTIPLE_CHOICE_INSTRUCTIONS
    assert row["question"]["criteria"] == {"A": "ignore", "B": "enforce", "C": "authoritarian", "D": "yell at", "E": "avoid"}
    assert row["label"] == row["target"] == "A"
    assert row["id"] == "extra-commonsense_qa-075e483d21c29a511267ef62bedc0461" == row["family"]


def test_openbookqa() -> None:
    raw = {"id": "7-980", "question_stem": "The sun is responsible for",
           "choices": {"text": ["puppies learning new tricks", "children growing up and getting old",
                                 "flowers wilting in a vase", "plants sprouting, blooming and wilting"],
                       "label": ["A", "B", "C", "D"]},
           "answerKey": "D"}
    row = extra.convert_openbookqa(raw)
    assert row["state"] == "The sun is responsible for"
    assert row["question"]["criteria"]["D"] == "plants sprouting, blooming and wilting"
    assert row["label"] == row["target"] == "D"


def test_arc_challenge_keeps_lettered_answer_key() -> None:
    raw = {"id": "Mercury_SC_415702",
           "question": "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?",
           "choices": {"text": ["dry palms", "wet palms", "palms covered with oil", "palms covered with lotion"],
                       "label": ["A", "B", "C", "D"]},
           "answerKey": "A"}
    row = extra.convert_arc_challenge(raw)
    assert row["state"] == raw["question"]
    assert row["label"] == row["target"] == "A"
    assert row["id"] == "extra-arc_challenge-Mercury_SC_415702"


def test_arc_easy_numeric_answer_key() -> None:
    raw = {"id": "numeric-demo", "question": "Which is heavier?",
           "choices": {"text": ["a feather", "a brick", "a leaf", "a balloon"], "label": ["1", "2", "3", "4"]},
           "answerKey": "2"}
    row = extra.convert_arc_easy(raw)
    assert row["question"]["criteria"] == {"1": "a feather", "2": "a brick", "3": "a leaf", "4": "a balloon"}
    assert row["label"] == row["target"] == "2"


def test_social_iqa() -> None:
    raw = {"context": "Cameron decided to have a barbecue and gathered her friends together.",
           "question": "How would Others feel as a result?",
           "answerA": "like attending", "answerB": "like staying home", "answerC": "a good friend to have",
           "label": "1"}
    row = extra.convert_social_iqa(0, raw)
    assert row["state"] == raw["context"]
    assert row["question"]["instructions"] == raw["question"]
    assert row["question"]["criteria"] == {"1": "like attending", "2": "like staying home", "3": "a good friend to have"}
    assert row["label"] == row["target"] == "1"
    assert row["id"] == "extra-social_iqa-0" == row["family"]


def test_cosmos_qa() -> None:
    raw = {"id": "3Q9SPIIRWJKVQ8244310E8TUS6YWAC##34V1S5K3GTZMDUBNBIGY93FLDOB690##A1S1K7134S2VUC##Blog_1044056##q1_a1##3XU9MCX6VQQG7YPLCSAFDPQNH4GR20",
           "context": ("Good Old War and person L : I saw both of these bands Wednesday night , and they both blew "
                       "me away . seriously . Good Old War is acoustic and makes me smile . I really can not help "
                       "but be happy when I listen to them ; I think it 's the fact that they seemed so happy "
                       "themselves when they played ."),
           "question": "In the future , will this person go to see other bands play ?",
           "answer0": "None of the above choices .",
           "answer1": "This person likes music and likes to see the show , they will see other bands play .",
           "answer2": "This person only likes Good Old War and Person L , no other bands .",
           "answer3": "Other Bands is not on tour and this person can not see them .",
           "label": 1}
    row = extra.convert_cosmos_qa(raw)
    assert row["state"] == raw["context"]
    assert row["question"]["instructions"] == raw["question"]
    assert row["question"]["criteria"]["1"] == raw["answer1"]
    assert row["label"] == row["target"] == "1"


def test_quartz() -> None:
    raw = {"id": "QRQA-10385-4",
           "question": ("John's town used to have lots of water, back when there were only a few hundred people. "
                        "However, now that the town holds several thousand people, the water availability is"),
           "choices": {"text": ["scarce", "plentiful"], "label": ["A", "B"]}, "answerKey": "A",
           "para": ("Many of the worlds people live with water scarcity, and that percentage will increase as "
                    "populations increase and climate changes.")}
    row = extra.convert_quartz(raw)
    assert row["state"] == raw["para"]
    assert row["question"]["instructions"] == raw["question"]
    assert row["question"]["criteria"] == {"A": "scarce", "B": "plentiful"}
    assert row["label"] == row["target"] == "A"


def test_qasc() -> None:
    raw = {"id": "3E7TUJ2EGCLQNOV1WEAJ2NN9ROPD9K", "question": "What type of water formation is formed by clouds?",
           "choices": {"text": ["pearls", "streams", "shells", "diamonds", "rain", "beads", "cooled", "liquid"],
                       "label": ["A", "B", "C", "D", "E", "F", "G", "H"]},
           "answerKey": "F", "fact1": "beads of water are formed by water vapor condensing",
           "fact2": "Clouds are made of water vapor.",
           "combinedfact": "Beads of water can be formed by clouds.",
           "formatted_question": "irrelevant, must not be used"}
    row = extra.convert_qasc(raw)
    assert row["state"] == "beads of water are formed by water vapor condensing Clouds are made of water vapor."
    assert row["question"]["instructions"] == raw["question"]
    assert len(row["question"]["criteria"]) == 8
    assert row["label"] == row["target"] == "F"
    assert row["question"]["criteria"]["F"] == "beads"


def test_truthful_qa_no_cap_needed_shuffles_options() -> None:
    raw = {"question": "What is the smallest country in the world that is at least one square mile in area?",
           "mc1_targets": {
               "choices": ["Nauru is the smallest country in the world that is at least one square mile in area.",
                           "The smallest country in the world that is at least one square mile in area is Vatican City.",
                           "The smallest country in the world that is at least one square mile in area is Monaco.",
                           "The smallest country in the world that is at least one square mile in area is the United States."],
               "labels": [1, 0, 0, 0]}}
    row = extra.convert_truthful_qa(0, raw)
    assert row["state"] == raw["question"]
    assert row["question"]["instructions"] == extra.TRUTHFULQA_INSTRUCTIONS
    assert len(row["question"]["criteria"]) == 4
    # The correct answer must not always land in the first slot.
    assert row["question"]["criteria"][row["label"]] == raw["mc1_targets"]["choices"][0]
    assert row["label"] == row["target"]


def test_truthful_qa_caps_at_ten_options_and_always_keeps_the_correct_one() -> None:
    choices = [f"choice {i}" for i in range(13)]
    labels = [0] * 13
    labels[7] = 1
    raw = {"question": "q", "mc1_targets": {"choices": choices, "labels": labels}}
    row = extra.convert_truthful_qa(3, raw)
    assert len(row["question"]["criteria"]) == extra.TRUTHFULQA_MAX_OPTIONS
    assert row["question"]["criteria"][row["label"]] == "choice 7"
    assert row["source"]["options_before_cap"] == 13
    # Deterministic: converting the same raw row again gives an identical result.
    again = extra.convert_truthful_qa(3, raw)
    assert row["question"]["criteria"] == again["question"]["criteria"] and row["label"] == again["label"]


def test_twitter_financial_sentiment_mapping() -> None:
    row = extra.convert_twitter_financial(0, {"text": "$BYND - JPMorgan reels in expectations on Beyond Meat", "label": "0"})
    assert row["label"] == row["target"] == "bearish"
    assert extra.convert_twitter_financial(1, {"text": "t", "label": "1"})["label"] == "bullish"
    assert extra.convert_twitter_financial(2, {"text": "t", "label": "2"})["label"] == "neutral"


def test_liar2_maps_six_truthfulness_levels_and_excludes_justification() -> None:
    raw = {"id": "13847", "label": "5",
           "statement": '90 percent of Americans "support universal background checks" for gun purchases.',
           "date": "October 2, 2017", "subject": "government regulation;polls and public opinion;guns",
           "speaker": "chris abele",
           "speaker_description": "Chris Abele is Milwaukee County Executive.",
           "state_info": "wisconsin", "true_counts": "1", "mostly_true_counts": "4", "half_true_counts": "5",
           "mostly_false_counts": "3", "false_counts": "5", "pants_on_fire_counts": "2", "context": "a tweet",
           "justification": "This must never appear in the state."}
    row = extra.convert_liar2(raw)
    assert row["state"] == (f'{raw["statement"]}\nSpeaker: chris abele\nContext: a tweet')
    assert "justification" not in row["state"] and "never appear" not in row["state"]
    assert row["label"] == row["target"] == "true"
    assert row["id"] == "extra-liar2-13847"


def test_liar2_all_levels() -> None:
    base = {"id": "x", "statement": "s", "speaker": "", "context": ""}
    for index, level in enumerate(extra.LIAR2_LEVELS):
        row = extra.convert_liar2({**base, "label": str(index)})
        assert row["label"] == level


def test_halueval_qa_noul() -> None:
    raw = {"knowledge": ("Arthur's Magazine (1844–1846) was an American literary periodical published in "
                        "Philadelphia in the 19th century.First for Women is a woman's magazine published by "
                        "Bauer Media Group in the USA."),
           "question": "Which magazine was started first Arthur's Magazine or First for Women?",
           "answer": "First for Women was started first.", "hallucination": "yes"}
    row = extra.convert_halueval_qa(0, raw)
    assert row["question"]["type"] == "noul"
    assert row["question"]["instructions"] == extra.HALUEVAL_INSTRUCTIONS
    assert row["state"] == f"Knowledge:\n{raw['knowledge']}\n\nQuestion: {raw['question']}\n\nResponse: {raw['answer']}"
    assert row["label"] is True and row["target"] is True
    no_row = extra.convert_halueval_qa(1, {**raw, "hallucination": "no"})
    assert no_row["label"] is False and no_row["target"] is False


def test_halueval_dialogue_noul() -> None:
    raw = {"knowledge": "Iron Man is starring Robert Downey Jr.Robert Downey Jr. starred in Zodiac (Crime Fiction Film)Zodiac (Crime Fiction Film) is starring Jake Gyllenhaal",
           "dialogue_history": "[Human]: Do you like Iron Man [Assistant]: Sure do! Robert Downey Jr. is a favorite. [Human]: Yes i like him too did you know he also was in Zodiac a crime fiction film. ",
           "response": "I'm not a fan of crime movies, but I did know that RDJ starred in Zodiac with Tom Hanks.",
           "hallucination": "yes"}
    row = extra.convert_halueval_dialogue(0, raw)
    assert row["state"] == (f"Knowledge:\n{raw['knowledge']}\n\nDialogue history: {raw['dialogue_history']}"
                            f"\n\nResponse: {raw['response']}")
    assert row["label"] is True


def test_halueval_summarization_noul() -> None:
    raw = {"document": "Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings Flight 9525.",
           "summary": "A video showing the final moments of Germanwings Flight 9525 has been recovered by investigators from the wreckage site.",
           "hallucination": "yes"}
    row = extra.convert_halueval_summarization(0, raw)
    assert row["state"] == f"Document:\n{raw['document']}\n\nSummary: {raw['summary']}"
    assert row["label"] is True


def test_halueval_rejects_unexpected_hallucination_value() -> None:
    with pytest.raises(ValueError, match="halueval_qa"):
        extra.convert_halueval_qa(0, {"knowledge": "k", "question": "q", "answer": "a", "hallucination": "maybe"})


def test_wikibio_one_row_per_sentence_shares_a_family() -> None:
    raw = {"wiki_bio_text": "Sir John Russell Reynolds was a British neurologist and physician.",
           "gpt3_sentences": ["John Russell Reynolds was an English lawyer.",
                               "He was born in London.",
                               "He was called to the bar in 1845."],
           "annotation": ["major_inaccurate", "accurate", "minor_inaccurate"],
           "wiki_bio_test_idx": 62464}
    rows = extra.convert_wikibio(raw)
    assert len(rows) == 3
    assert all(row["family"] == "extra-wikibio-62464" for row in rows)
    assert [row["id"] for row in rows] == ["extra-wikibio-62464-0", "extra-wikibio-62464-1", "extra-wikibio-62464-2"]
    assert [row["label"] for row in rows] == [True, False, True]
    assert rows[0]["state"] == (f"Reference biography:\n{raw['wiki_bio_text']}\n\n"
                                 f"Sentence to check: {raw['gpt3_sentences'][0]}")
    assert rows[0]["question"]["instructions"] == extra.WIKIBIO_INSTRUCTIONS


def test_wikibio_mismatched_sentence_and_annotation_counts_raises() -> None:
    with pytest.raises(ValueError, match="62464"):
        extra.convert_wikibio({"wiki_bio_text": "t", "gpt3_sentences": ["a", "b"], "annotation": ["accurate"],
                               "wiki_bio_test_idx": 62464})


ALL_CONVERTED_ROWS = [
    extra.convert_snli(0, {"premise": "p", "hypothesis": "h", "label": 0}),
    extra.convert_commonsense_qa({"id": "c1", "question": "q", "choices": {"label": ["A", "B"], "text": ["x", "y"]}, "answerKey": "A"}),
    extra.convert_openbookqa({"id": "o1", "question_stem": "q", "choices": {"label": ["A", "B"], "text": ["x", "y"]}, "answerKey": "B"}),
    extra.convert_arc_challenge({"id": "a1", "question": "q", "choices": {"label": ["1", "2"], "text": ["x", "y"]}, "answerKey": "1"}),
    extra.convert_arc_easy({"id": "a2", "question": "q", "choices": {"label": ["A", "B"], "text": ["x", "y"]}, "answerKey": "A"}),
    extra.convert_social_iqa(0, {"context": "c", "question": "q", "answerA": "x", "answerB": "y", "answerC": "z", "label": "2"}),
    extra.convert_cosmos_qa({"id": "co1", "context": "c", "question": "q", "answer0": "w", "answer1": "x", "answer2": "y", "answer3": "z", "label": 3}),
    extra.convert_quartz({"id": "qz1", "question": "q", "para": "p", "choices": {"label": ["A", "B"], "text": ["x", "y"]}, "answerKey": "B"}),
    extra.convert_qasc({"id": "qs1", "question": "q", "fact1": "f1", "fact2": "f2",
                        "choices": {"label": [chr(65 + i) for i in range(8)], "text": [str(i) for i in range(8)]}, "answerKey": "C"}),
    extra.convert_truthful_qa(0, {"question": "q", "mc1_targets": {"choices": ["a", "b", "c"], "labels": [0, 1, 0]}}),
    extra.convert_twitter_financial(0, {"text": "t", "label": "1"}),
    extra.convert_liar2({"id": "l1", "statement": "s", "speaker": "", "context": "", "label": "3"}),
    extra.convert_halueval_qa(0, {"knowledge": "k", "question": "q", "answer": "a", "hallucination": "no"}),
    extra.convert_halueval_dialogue(0, {"knowledge": "k", "dialogue_history": "d", "response": "r", "hallucination": "yes"}),
    extra.convert_halueval_summarization(0, {"document": "doc", "summary": "s", "hallucination": "no"}),
    *extra.convert_wikibio({"wiki_bio_text": "t", "gpt3_sentences": ["s1"], "annotation": ["accurate"], "wiki_bio_test_idx": 1}),
]


def test_every_converter_produces_a_label_that_is_a_valid_option_key() -> None:
    validate(ALL_CONVERTED_ROWS)
    for row in ALL_CONVERTED_ROWS:
        if row["question"]["type"] == "choice":
            assert row["label"] in row["question"]["criteria"]
        else:
            assert type(row["label"]) is bool


def test_sample_caps_at_five_thousand_and_is_deterministic() -> None:
    rows = [extra.convert_twitter_financial(i, {"text": f"t{i}", "label": "0"}) for i in range(6000)]
    sampled = extra.sample(rows, "twitter_financial")
    assert len(sampled) == extra.CAP == 5000
    again = extra.sample(rows, "twitter_financial")
    assert [row["id"] for row in sampled] == [row["id"] for row in again]
    assert [row["id"] for row in sampled] == sorted(row["id"] for row in sampled)
    # A different dataset name reshuffles differently, so the kept subset need not match.
    other = extra.sample(rows, "some_other_dataset")
    assert [row["id"] for row in other] != [row["id"] for row in sampled]


def test_sample_keeps_everything_when_under_the_cap() -> None:
    rows = [extra.convert_twitter_financial(i, {"text": f"t{i}", "label": "0"}) for i in range(10)]
    sampled = extra.sample(rows, "twitter_financial")
    assert len(sampled) == 10
    assert {row["id"] for row in sampled} == {row["id"] for row in rows}


def test_benchmark_training_splits_use_the_panel_format_under_their_own_names() -> None:
    raw = {"id": "77", "query": "Summarize the news:", "context": "The mayor opened a bridge.", "output": "The mayor opened a tunnel.",
           "task_type": "Summary", "hallucination_labels": '[{"text": "tunnel"}]'}
    row = extra.convert_ragtruth_train(raw)
    assert row["id"] == "extra-ragtruth_train-77" and row["suite"] == "ragtruth_train"
    assert row["state"] == {"task": "Summary", "instruction": "Summarize the news:", "source": "The mayor opened a bridge.",
                            "response": "The mayor opened a tunnel."}
    assert row["label"] is True and row["source"]["split"] == "train"
    wino = extra.convert_winogrande_train(3, {"sentence": "Ann beat Bo so _ won.", "option1": "Ann", "option2": "Bo", "answer": "1"})
    assert wino["id"] == "extra-winogrande_train-3" and wino["question"]["criteria"] == {"1": "Ann", "2": "Bo"}
    assert wino["label"] == "1" and wino["source"]["config"] == "winogrande_xl"
