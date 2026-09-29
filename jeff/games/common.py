"""Shared pieces for the game tests: a client that asks a Jeff server (or any server with the same decision API) for each move, and
an episode runner that records scores, decisions and time per decision.

Every game exposes the same small interface (see Game): the state as plain observations, one choice question whose
options are the legal moves, and a step function. A player is any function from (state, question) to one option key."""

import json
import random
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import httpx

Player = Callable[[dict, dict], str]


class Game(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...
    def observe(self) -> tuple[dict, dict]: ...  # (state, choice question whose criteria are the legal moves)
    def step(self, move: str) -> None: ...
    @property
    def done(self) -> bool: ...
    def result(self) -> dict[str, float]: ...  # this episode's measurements; "score" is the headline number


class JeffClient:
    """Ask a Jeff server for one decision. Errors are raised, never replaced by a default move."""

    def __init__(self, url: str, model: str = "jeff-latest") -> None:
        self.url = url.rstrip("/") + "/v1/systemone"
        self.model = model
        self.http = httpx.Client(timeout=30)
        self.confidences: list[float] = []
        self.last: dict | None = None  # the latest answer, for video captions

    def ask(self, state: dict, questions: dict[str, dict]) -> dict[str, dict]:
        response = self.http.post(self.url, json={"model": self.model, "state": state, "questions": questions})
        if response.status_code != 200:
            raise RuntimeError(f"Jeff server answered {response.status_code}: {response.text[:500]}")
        return response.json()["answers"]

    def player(self) -> Player:
        def choose(state: dict, question: dict) -> str:
            answer = self.ask(state, {"move": question})["move"]
            self.confidences.append(answer["confidence"])
            self.last = answer
            return str(answer["choice"])
        return choose


def random_player(seed: int) -> Player:
    rng = random.Random(seed)
    return lambda state, question: rng.choice(sorted(question["criteria"]))


def run(game: Game, player: Player, seeds: list[int], max_steps: int, record: Path | None = None,
        title: str = "", client: "JeffClient | None" = None) -> dict:
    """Play one episode per seed; an episode also ends after max_steps decisions. With `record`, the first episode is
    also written as a video (one frame per decision, captioned with the options and the choice)."""
    from jeff.games import video

    episodes, seconds, frames = [], [], []
    for number, seed in enumerate(seeds):
        game.reset(seed)
        steps = 0
        while not game.done and steps < max_steps:
            state, question = game.observe()
            began = time.perf_counter()
            move = player(state, question)
            seconds.append(time.perf_counter() - began)
            if move not in question["criteria"]:
                raise ValueError(f"{game.name}: the player chose {move!r}, which is not one of {sorted(question['criteria'])}")
            if record is not None and number == 0:
                probabilities = client.last["probabilities"] if client is not None and client.last else None
                score = game.result()["score"]
                frames.append(video.frame(game.render(), f"{title} · turn {steps + 1} · score {score:g}", question["criteria"], move,
                                          probabilities, 1000 * seconds[-1] if client is not None else None))
            game.step(move)
            steps += 1
        episodes.append({"seed": seed, "steps": steps, **game.result()})
    if record is not None:
        video.write(frames, record, fps=4)
    scores = [e["score"] for e in episodes]
    return {"game": game.name, "episodes": episodes, "mean_score": statistics.fmean(scores),
            "stdev_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "decisions": len(seconds), "median_ms_per_decision": 1000 * statistics.median(seconds) if seconds else None}


def save(result: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
