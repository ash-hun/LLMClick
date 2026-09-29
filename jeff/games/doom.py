"""Doom (ViZDoom's defend_the_center), played exactly as in jev-plays-doom, so the scores compare with Jev's.

Adapted from https://github.com/tirukovelamanoj/jev-plays-doom (commit 496f0e1), MIT License, Copyright (c) 2026 Manoj
Tirukovela: the scenario settings, the text description of the game state, the two sets of option texts ("rule", which
hands the model the 8-degree aiming threshold, and "intent", which only says what each button does), the extra
"in_danger" yes/no question asked in the same call, 4 game tics per decision, and seed 1234. Their published results
over 20 episodes on that seed: random 0.75 kills, hand-coded 6.55, Jev given the rule 6.55, Jev judging for itself -0.60.
The player never sees pixels: it gets the state as text and presses one of three buttons."""

import math
import os
import random
import statistics
import time
from collections.abc import Callable

BUTTONS = ["TURN_LEFT", "TURN_RIGHT", "ATTACK"]
CRITERIA_RULE = {
    "ATTACK": "Fire. Correct whenever the nearest monster's bearing is between "
              "-8 and +8, i.e. already lined up. Firing is how you score; do it "
              "the moment you are aligned rather than adjusting further.",
    "TURN_LEFT": "Rotate left. Correct only when the nearest monster's bearing is "
                 "GREATER than +8. Do not pick this if the bearing is between -8 "
                 "and +8; fire instead.",
    "TURN_RIGHT": "Rotate right. Correct only when the nearest monster's bearing "
                  "is LESS than -8. Do not pick this if the bearing is between -8 "
                  "and +8; fire instead.",
}
CRITERIA_INTENT = {
    "ATTACK": "Fire your weapon down your line of sight. This is the only way to "
              "score, but a shot that is not lined up hits nothing and wastes a "
              "bullet you cannot spare.",
    "TURN_LEFT": "Rotate left. This swings your line of sight toward monsters that "
                 "are to your left, and away from ones on your right.",
    "TURN_RIGHT": "Rotate right. This swings your line of sight toward monsters "
                  "that are to your right, and away from ones on your left.",
}
CRITERIA = {"rule": CRITERIA_RULE, "judgement": CRITERIA_INTENT}


def outcomes(desc: dict) -> dict[str, str]:
    """What each button would do right now, in words (ours, not the harness's; the model still chooses)."""
    monsters = desc["monsters_nearest_first"]
    if not monsters:
        return {"ATTACK": "Fire: no monster in sight; the shot would be wasted.",
                "TURN_LEFT": "Turn left: look around for monsters.", "TURN_RIGHT": "Turn right: look around for monsters."}
    bearing = monsters[0]["bearing_deg"]
    if abs(bearing) <= 8:
        return {"ATTACK": "Fire: the nearest monster is lined up; the shot would hit it.",
                "TURN_LEFT": "Turn left: the nearest monster is already lined up; turning would lose your aim.",
                "TURN_RIGHT": "Turn right: the nearest monster is already lined up; turning would lose your aim."}
    side = "left" if bearing > 0 else "right"
    towards = f"turns towards the nearest monster, {abs(bearing)} degrees to your {side}."
    away = f"turns away from the nearest monster, which is to your {side}."
    return {"ATTACK": "Fire: nothing is lined up; the shot would miss and waste a bullet.",
            "TURN_LEFT": f"Turn left: {towards if side == 'left' else away}",
            "TURN_RIGHT": f"Turn right: {towards if side == 'right' else away}"}
INSTRUCTIONS = ("You are fighting for your life in Doom, surrounded by monsters. Pick the single best button to press "
                "right now.")
MONSTERS = ("Demon", "Cacodemon", "ZombieMan", "ShotgunGuy", "HellKnight", "Imp", "Revenant")


def describe(state) -> dict:
    hp, ammo, px, py, ang = state.game_variables
    monsters = []
    for o in state.objects:
        if o.name in MONSTERS:
            dx, dy = o.position_x - px, o.position_y - py
            bearing = (math.degrees(math.atan2(dy, dx)) - ang + 540) % 360 - 180
            monsters.append({"kind": o.name, "distance": round(math.hypot(dx, dy)), "bearing_deg": round(bearing)})
    monsters.sort(key=lambda m: m["distance"])
    return {"health": int(hp), "ammo": int(ammo), "facing_deg": round(ang), "monsters_nearest_first": monsters[:6],
            "note": "bearing 0 = dead ahead. POSITIVE bearing means the monster is to "
                    "your LEFT (turn left to face it); NEGATIVE means to your RIGHT."}


def where(bearing: int) -> str:
    if abs(bearing) <= 8:
        return "straight ahead"
    side = "left" if bearing > 0 else "right"
    return f"a little to your {side}" if abs(bearing) <= 30 else f"well to your {side}"


def situation(desc: dict) -> tuple[dict, dict[str, str]]:
    """A short state and, per button, where the nearest monster is, in words, without saying what is right or wrong."""
    monsters = desc["monsters_nearest_first"]
    state = {"health": desc["health"], "ammo": desc["ammo"], "monsters_in_view": len(monsters)}
    if not monsters:
        return state, {"ATTACK": "Fire: no monster in view.", "TURN_LEFT": "Turn left: no monster in view.",
                       "TURN_RIGHT": "Turn right: no monster in view."}
    bearing = monsters[0]["bearing_deg"]
    place = where(bearing)
    turn = lambda side: ("the nearest monster is straight ahead" if abs(bearing) <= 8 else  # noqa: E731
                         f"towards the nearest monster" if (bearing > 0) == (side == "left") else "away from the nearest monster")
    return state, {"ATTACK": f"Fire: the nearest monster is {place}.",
                   "TURN_LEFT": f"Turn left: {turn('left')}.", "TURN_RIGHT": f"Turn right: {turn('right')}."}


def rule_player(desc: dict) -> str:
    monsters = desc["monsters_nearest_first"]
    if not monsters:
        return "TURN_RIGHT"
    bearing = monsters[0]["bearing_deg"]
    if abs(bearing) <= 8:
        return "ATTACK"
    return "TURN_LEFT" if bearing > 0 else "TURN_RIGHT"


def jeff_player(client, criteria: str) -> Callable[[dict], str]:
    """Ask for the action. Jev's own wordings ("rule", "judgement") also ask the harness's in_danger question in the same
    call, exactly as jev-plays-doom does; our wordings skip it, since it takes a second model pass and nothing uses it."""
    def choose(desc: dict) -> str:
        state, options = situation(desc) if criteria == "situation" else (desc, outcomes(desc) if criteria == "outcomes" else CRITERIA[criteria])
        client.last_options = options
        questions = {"action": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": options}}
        if criteria in ("rule", "judgement"):  # the harness's exact request also asks this; nothing uses the answer
            questions["in_danger"] = {"type": "noul", "instructions": "Is a monster close enough to hurt you within the next second?"}
        answers = client.ask(state, questions)
        client.confidences.append(answers["action"]["confidence"])
        client.last = answers["action"]
        return str(answers["action"]["choice"])
    return choose


def run(player: Callable[[dict], str], episodes: int = 20, seed: int = 1234, tics: int = 4, record=None, title: str = "",
        client=None) -> dict:
    """Play the episodes; with `record` (a path), the first episode is also written as a video of the real game screen,
    captioned with each decision, at real-time speed (35 game tics per second, one decision every `tics`)."""
    import vizdoom as vzd
    from PIL import Image

    from jeff.games import video

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
    game.set_window_visible(False)
    game.set_objects_info_enabled(True)
    game.set_available_game_variables([vzd.GameVariable.HEALTH, vzd.GameVariable.AMMO2, vzd.GameVariable.POSITION_X,
                                       vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE])
    game.set_seed(seed)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    scores, seconds, frames = [], [], []
    for number in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            if state is None:
                break
            began = time.perf_counter()
            pick = player(describe(state))
            seconds.append(time.perf_counter() - began)
            if pick not in BUTTONS:
                raise ValueError(f"doom: the player chose {pick!r}, not one of {BUTTONS}")
            if record is not None and number == 0:
                options = getattr(client, "last_options", None) or {button: "" for button in BUTTONS}
                probabilities = client.last["probabilities"] if client is not None and client.last else None
                frames.append(video.frame(Image.fromarray(state.screen_buffer), f"{title} · kills so far {game.get_total_reward():.0f}",
                                          options, pick, probabilities, 1000 * seconds[-1] if client is not None else None))
            game.make_action([1 if button == pick else 0 for button in BUTTONS], tics)
        scores.append(game.get_total_reward())
    game.close()
    if record is not None:
        video.write(frames, record, fps=35 / tics)
    return {"game": "doom", "episodes": [{"episode": i + 1, "score": s} for i, s in enumerate(scores)],
            "mean_score": statistics.fmean(scores), "stdev_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "decisions": len(seconds), "median_ms_per_decision": 1000 * statistics.median(seconds) if seconds else None}


def random_player(seed: int) -> Callable[[dict], str]:
    rng = random.Random(seed)
    return lambda desc: rng.choice(BUTTONS)
