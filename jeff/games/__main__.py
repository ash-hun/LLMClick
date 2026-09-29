"""Zero-shot game tests for Jeff: Frogger, Pac-Man and Doom, each against random play and a hand-coded rule bot.

    python -m jeff.games --game frogger --player jeff --url http://127.0.0.1:8766 --criteria rule --out runs/games/x.json

--criteria rule gives each option the condition under which it is the right move (as the Doom harness does for Jev);
--criteria judgement only says what each move does. Episodes use seeds SEED, SEED+1, ... (Doom: one seed for the run)."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from jeff.games import common, doom, frogger, pacman

MAX_STEPS = {"frogger": 300, "pacman": 600}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", choices=("frogger", "pacman", "doom"), required=True)
    parser.add_argument("--player", choices=("random", "rule", "jeff"), required=True)
    parser.add_argument("--criteria", choices=("rule", "judgement", "outcomes", "situation", "described", "space", "space2", "distance", "zones"), default="rule")
    parser.add_argument("--url", help="Jeff server for --player jeff, e.g. http://127.0.0.1:8766")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--note", help="One line on the strategy of this run, shown on the dashboard")
    parser.add_argument("--video", action="store_true", help="Also record the first episode as runs/games/videos/<out name>.mp4")
    args = parser.parse_args()
    if args.player == "jeff" and not args.url:
        raise SystemExit("--player jeff needs --url")
    client = common.JeffClient(args.url) if args.player == "jeff" else None
    record = args.out.parent / "videos" / f"{args.out.stem}.mp4" if args.video else None
    if record is not None and record.exists():
        raise FileExistsError(f"Refusing to overwrite {record}")
    who = (Path(client.http.get(args.url.rstrip("/") + "/health").json()["checkpoint"]).name if client else
           {"random": "random moves", "rule": "rule bot"}[args.player])
    title = f"{args.game} · {who}" + (f" · {args.criteria} wording" if client else "")
    if args.game == "doom":
        player = (doom.random_player(args.seed) if args.player == "random" else doom.rule_player if args.player == "rule"
                  else doom.jeff_player(client, args.criteria))
        result = doom.run(player, episodes=args.episodes, seed=args.seed, record=record, title=title, client=client)
    else:
        game = (frogger.Frogger if args.game == "frogger" else pacman.PacMan)(args.criteria)
        module = frogger if args.game == "frogger" else pacman
        player = (common.random_player(args.seed) if args.player == "random" else module.rule_player if args.player == "rule"
                  else client.player())
        result = common.run(game, player, list(range(args.seed, args.seed + args.episodes)), MAX_STEPS[args.game],
                            record=record, title=title, client=client)
    code = Path(__file__).parent / f"{args.game}.py"
    result.update({"player": args.player, "criteria": args.criteria if args.player == "jeff" else None,
                   "url": args.url, "seed": args.seed, "created": datetime.now(timezone.utc).isoformat(),
                   "game_code_sha256": hashlib.sha256(code.read_bytes()).hexdigest(), "note": args.note,
                   "video": record.name if record is not None else None})
    if client is not None:
        result["server"] = client.http.get(args.url.rstrip("/") + "/health").json()
        result["mean_confidence"] = sum(client.confidences) / len(client.confidences)
    common.save(result, args.out)
    print(json.dumps({k: result[k] for k in ("game", "player", "criteria", "mean_score", "stdev_score", "decisions",
                                             "median_ms_per_decision")}))


if __name__ == "__main__":
    main()
