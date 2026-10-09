"""`data_synthetic` end to end with a deterministic fake teacher: every stage is cached, resumable and the same twice."""

import io
import json
from pathlib import Path
from typing import Any

import pytest

from core import pipeline
from core.progress import StateProgress
from data.evolve import EVOLVERS, evolve_prompt, failed
from data.respond import BudgetExceeded
from data.rows import key
from data.select import assemble
from data.teachers import TEACHERS, Completion, Teacher
from data.verify import majority_keys, sorry_short


class Fake(Teacher):
    """Answers from the request text alone, so the same request always gets the same answer; counts every call."""
    name = "fake"
    calls: list[list[dict[str, str]]] = []
    fail_after: int | None = None

    def complete(self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float, seed: int) -> Completion:
        if Fake.fail_after is not None and len(Fake.calls) >= Fake.fail_after:
            raise ConnectionError("the teacher went away")
        Fake.calls.append(messages)
        content = messages[-1]["content"]
        if "questions or requests related to" in content:
            topic = content.split("related to ")[1].split("?")[0]
            text = "\n".join(f"{i + 1}. What is {topic} question {i + 1}?" for i in range(3))
        elif content.startswith("Rewrite this question"):
            text = content.rsplit("\n\n", 1)[1] + " (specifically)"
        elif "Prompt Rewriter" in content:
            text = content.split("#The Given Prompt#:\n")[1].split("\n#Rewritten Prompt#")[0] + " Explain each step."
        elif "Prompt Creator" in content:
            text = "A rarer prompt in the same domain."
        elif content.startswith("Does the second prompt add"):
            text = "yes"
        elif "Reply with a single integer score" in content:
            text = "9" if "step" in content else "4"
        elif "What is 2 plus 2" in content:
            text = "2 plus 2 is 4. Final answer: 4" if seed % 2 == 0 else "I think 5"
        elif "Persona:" in content:
            text = f"Please help me with {content.split('about ')[1].split('.')[0]} as {content.split('Persona: ')[1].split(chr(10))[0]}."
        elif "unanswerable" in content:
            text = "Sorry, I cannot help with that."
        else:
            text = f"Answer {seed}: {content[:40]}"
        return Completion(text, len(content.split()), len(text.split()), self.price[1] * len(text.split()) / 1_000_000)


@pytest.fixture(autouse=True)
def fake_teacher() -> Any:
    if "fake" not in TEACHERS:
        TEACHERS.register("fake")(Fake)
    Fake.calls, Fake.fail_after = [], None
    yield
    Fake.fail_after = None


def config(out: Path, **keys: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "pipeline": {"recipe": "data_synthetic", "name": "syn", "seed": 3, "output_dir": str(out)},
        "teacher": {"name": "fake", "model": "fake-1", "price": [1.0, 2.0]},
        "seeds": [{"kind": "topic", "values": ["tides", "yeast", "tides"]}],
        "prompts": {"generator": "topic_questions", "params": {"n": 2}},
        "respond": {"answers_per_question": 2, "max_tokens": 64},
        "verify": {"schema_recipe": "llm_sft"},
        "select": {"assemble": "sft", "dedup": {"near": None}},
    }
    return {**base, **keys}


def test_seeds_prompts_answers_rows_and_a_rerun_builds_nothing(tmp_path: Path) -> None:
    progress = StateProgress()
    result = pipeline.build(config(tmp_path), progress).run()
    stages = result["stages"]
    assert stages["seeds"]["count"] == 2 and stages["seeds"]["kinds"] == {"topic": 2}  # the repeated seed is one seed
    assert stages["prompts"]["count"] == 4 and stages["prompts"]["calls"] == 2 * (1 + 2)  # per seed: one list, two refinements
    assert stages["respond"]["count"] == 8 and stages["respond"]["calls"] == 8
    assert stages["verify"]["count"] == 8 and stages["verify"]["rejected"] == 0
    assert stages["select"]["count"] == 4 and stages["select"]["counts"]["exact_duplicate"] == 0
    rows = [json.loads(line) for line in Path(stages["select"]["rows"]).read_text().splitlines()]
    assert all(row["messages"][0]["role"] == "user" and row["messages"][1]["role"] == "assistant" for row in rows)
    assert "(specifically)" in rows[0]["messages"][0]["content"]
    manifest = json.loads(Path(stages["select"]["manifest"]).read_text())
    assert manifest["teacher"]["model"] == "fake-1" and manifest["cost_usd"] == pytest.approx(stages["select"]["cost_usd"]) and manifest["cost_usd"] > 0
    assert "evolve" not in stages and progress.snapshot()["stages"]["select"] == "done"

    Fake.calls = []
    again = StateProgress()
    second = pipeline.build(config(tmp_path), again).run()
    assert json.dumps(second) == json.dumps(result) and Fake.calls == []
    assert set(again.snapshot()["stages"].values()) == {"cached"}


def test_a_crash_mid_respond_resumes_without_repeating_calls(tmp_path: Path) -> None:
    body = config(tmp_path)
    Fake.fail_after = 6 + 3  # the prompt stage's 6 calls, then three answers
    with pytest.raises(ConnectionError):
        pipeline.build(body).run()
    assert len(Fake.calls) == 9
    Fake.calls, Fake.fail_after = [], None
    result = pipeline.build(body).run()
    assert len(Fake.calls) == 5 and result["stages"]["respond"]["calls"] == 8  # the three answered before the crash were kept


def test_the_budget_stops_before_the_call_that_would_exceed_it(tmp_path: Path) -> None:
    body = config(tmp_path, budget={"max_usd": 0.00002})
    with pytest.raises(BudgetExceeded, match="run again"):
        pipeline.build(body).run()
    answered = len([c for c in Fake.calls if "What is" in c[-1]["content"] and "Rewrite" not in c[-1]["content"]])
    assert 0 < answered < 8
    Fake.calls = []
    result = pipeline.build(config(tmp_path, budget={"max_usd": 1.0})).run()  # the budget is not part of the identity
    assert result["stages"]["respond"]["calls"] == 8 and len(Fake.calls) == 8 - answered


def test_evolve_operators_elimination_and_judge_gain(tmp_path: Path) -> None:
    assert "Please add one more constraints" in evolve_prompt("constraints", "Write a poem.")
    assert evolve_prompt("breadth", "Write a poem.").startswith("I want you act as a Prompt Creator")
    assert failed("a", "The #Rewritten Prompt# is ...") == "scaffolding" and failed("a", " .. ") == "empty"
    assert failed("Write a poem.", "write a poem.") == "unchanged" and failed("a", "b") is None
    assert "easier" in EVOLVERS and "code_misdirection" in EVOLVERS
    body = config(tmp_path, evolve={"rounds": 2, "operators": {"constraints": 1, "breadth": 1}, "judge_gain": True})
    result = pipeline.build(body).run()
    evolve = result["stages"]["evolve"]
    dropped = evolve["dropped"]
    assert evolve["originals"] == 4 and evolve["count"] == 12 - sum(dropped.values()) and dropped["scaffolding"] == 0 and dropped["no_gain"] == 0
    rows = [json.loads(line) for line in Path(evolve["rows"]).read_text().splitlines()]
    assert {row.get("round") for row in rows} == {None, 1, 2} and all(row["operator"] in {"constraints", "breadth"} for row in rows if row.get("round"))
    assert result["stages"]["respond"]["prompts"] == evolve["count"]  # answers go to originals and evolved prompts alike
    with pytest.raises(ValueError, match="Unknown evolve operators"):
        pipeline.build(config(tmp_path, evolve={"operators": {"nope": 1}}))


def test_verify_rules_reward_majority_and_judge(tmp_path: Path) -> None:
    assert sorry_short("Sorry, no.", 80) and not sorry_short("Sorry " + "word " * 100, 80)
    rows = [{"prompt_id": "p", "text": "so 4"}, {"prompt_id": "p", "text": "it is 4"}, {"prompt_id": "p", "text": "5"}]
    assert majority_keys(rows, "last_number") == {"p": repr(4.0)}
    body = config(tmp_path, seeds=[{"kind": "instruction", "values": ["What is 2 plus 2?", "unanswerable request"]}],
                  prompts={"generator": "passthrough"}, respond={"answers_per_question": 4},
                  verify={"schema_recipe": "llm_sft", "majority": "last_number"})
    result = pipeline.build(body).run()
    verify = result["stages"]["verify"]
    assert verify["reasons"] == {"sorry": 4, "majority": 2} and verify["count"] == 2  # seeds 0 and 2 answered 4, 1 and 3 said 5
    judged = config(tmp_path, seeds=[{"kind": "instruction", "values": ["Describe tides."]}], prompts={"generator": "passthrough"},
                    respond={"answers_per_question": 2}, verify={"judge": {"rubric": "Rate it.", "min_score": 6}},
                    select={"assemble": "dpo", "margin": 1.0, "dedup": {"near": None}})
    result = pipeline.build(judged).run()
    assert result["stages"]["verify"]["count"] == 0  # the fake judge gives 4 to answers without "step"
    rewarded = config(tmp_path, seeds=[{"kind": "instruction", "source": {"name": "local_jsonl", "path": "samples/llm_grpo.jsonl", "limit": 3}, "field": "prompt"}],
                      prompts={"generator": "passthrough", "params": {"answer_field": "answer"}}, respond={"answers_per_question": 1},
                      verify={"reward": {"name": "contains"}}, select={"assemble": "grpo", "dedup": {"near": None}})
    result = pipeline.build(rewarded).run()
    assert result["stages"]["verify"]["reasons"] == {"reward": 3}  # the fake never answers the sums


def test_select_dedup_leak_and_assembly(tmp_path: Path) -> None:
    group = [{"prompt_id": "p", "sample": 0, "instruction": "q", "text": "weak", "score": 4.0},
             {"prompt_id": "p", "sample": 1, "instruction": "q", "text": "strong", "score": 9.0}]
    assert assemble("sft", group, 1.0) == {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "strong"}]}
    assert assemble("dpo", group, 1.0) == {"prompt": "q", "chosen": "strong", "rejected": "weak"}
    assert assemble("dpo", group, 6.0) is None and assemble("grpo", group, 1.0) is None
    leak = tmp_path / "eval.jsonl"
    leak.write_text(json.dumps({"messages": [{"role": "user", "content": "What is tides question 1? (specifically)"}]}) + "\n")
    body = config(tmp_path, respond={"answers_per_question": 1},
                  select={"assemble": "sft", "dedup": {"near": None}, "leak": {"against": [{"name": "local_jsonl", "path": str(leak)}], "threshold": 0.5, "ngram": 3}})
    counts = pipeline.build(body).run()["stages"]["select"]["counts"]
    assert counts["leaked"] == 1 and counts["rows"] == 3 and counts["near_duplicate"] == 0
    near = config(tmp_path, respond={"answers_per_question": 1}, select={"assemble": "sft", "dedup": {"near": 0.5, "ngram": 2}})
    counts = pipeline.build(near).run()["stages"]["select"]["counts"]
    assert counts["near_duplicate"] >= 1 and counts["rows"] + counts["near_duplicate"] == 4  # "tides question 1" vs "tides question 2"
    assert key("a", 1) == key("a", 1) and key("a", 1) != key("a", 2)


def test_the_shipped_config_validates_and_the_api_lists_the_recipe() -> None:
    from fastapi.testclient import TestClient

    from core.api.app import app
    from core.settings import get_settings

    get_settings.cache_clear()
    built = pipeline.load("configs/data/synthetic.yaml")
    assert built.plan() == ["seeds", "prompts", "evolve", "respond", "verify", "select"]
    recipes = TestClient(app).get("/api/system/recipes").json()
    assert set(recipes["data_synthetic"]["teacher"]) >= {"anthropic", "openai", "ollama", "local"}


def test_ollama_and_openai_teachers_speak_their_protocols(monkeypatch: pytest.MonkeyPatch) -> None:
    from data import teachers

    sent: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def fake_post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
        sent.append((url, body, headers))
        if "/api/chat" in url:
            return {"message": {"content": "ollama says hi"}, "prompt_eval_count": 7, "eval_count": 3}
        return {"choices": [{"message": {"content": "openai says hi"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}

    monkeypatch.setattr(teachers, "post_json", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    ollama = teachers.Ollama({"model": "qwen3.5:0.8b", "keep_alive": "5m"})
    done = ollama.complete([{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.5, seed=1)
    assert done == Completion("ollama says hi", 7, 3, 0.0)
    assert sent[0][0] == "http://localhost:11434/api/chat" and sent[0][1]["options"] == {"temperature": 0.5, "num_predict": 10, "seed": 1}
    openai = teachers.OpenAIFormat({"model": "gpt-x", "base_url": "http://vllm:8000/v1/", "price": [1, 2]})
    done = openai.complete([{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.5, seed=1)
    assert done.text == "openai says hi" and done.cost_usd == pytest.approx((7 * 1 + 3 * 2) / 1_000_000)
    assert sent[1][0] == "http://vllm:8000/v1/chat/completions" and sent[1][2]["authorization"] == "Bearer k"
    assert openai.identity() == {"name": "openai", "model": "gpt-x"}  # keys and prices are not identity


def test_the_anthropic_teacher_uses_the_sdk_and_prices_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from data import teachers

    class Usage:
        input_tokens, output_tokens = 100, 50

    class Block:
        type, text = "text", "claude says hi"

    class Response:
        stop_reason, content, usage = "end_turn", [Block()], Usage()

    class Messages:
        def create(self, **keys: Any) -> Response:
            assert keys["model"] == "claude-opus-5-5" and keys["system"] == "be brief" and keys["messages"][0]["role"] == "user"
            return Response()

    class Client:
        messages = Messages()

    teacher = teachers.Claude({})
    teacher.client = Client()
    done = teacher.complete([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], max_tokens=10, temperature=0.5, seed=1)
    assert done.text == "claude says hi" and done.cost_usd == pytest.approx(100 * 4 / 1_000_000 + 50 * 20 / 1_000_000)
    monkeypatch.setattr(io, "BytesIO", io.BytesIO)  # keep the import used
