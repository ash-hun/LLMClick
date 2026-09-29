"""A small Pac-Man, deterministic for a given seed, played one decision per turn.

One maze without dead ends (as in the arcade game), pellets on every open square, two ghosts, no power pellets. Each turn Pac-Man moves one square, then each ghost
moves one square: towards Pac-Man along the corridors with probability 0.9 (0.7 for the second ghost), otherwise
in a random direction that does not reverse. Meeting a ghost (or passing through it) costs a life; Pac-Man and the ghosts
then return to their starting squares. Three lives. Eating every pellet clears the maze and ends the episode.

The state is the same for both question versions: the maze as text, plus for each open direction the number of steps to
the nearest pellet and to the nearest ghost when leaving that way (the facts the rule version refers to, like Doom's
bearing)."""

import random
from collections import deque

MAZE = [
    "###############",
    "#......#......#",
    "#.###..#..###.#",
    "#.............#",
    "#.##.#####.##.#",
    "#....#...#....#",
    "####.#.#.#.####",
    "#......#......#",
    "#.###.....###.#",
    "#...#.###.#...#",
    "#.#.#.....#.#.#",
    "#.............#",
    "###############",
]
START, GHOST_STARTS = (11, 7), [(5, 7), (5, 6)]
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
REVERSE = {"up": "down", "down": "up", "left": "right", "right": "left"}
DANGER = 3  # steps

INSTRUCTIONS = ("You are Pac-Man in a maze. Eat all the pellets and avoid the ghosts: touching a ghost costs a life. Each "
                "turn you move one square along a corridor; then each ghost moves one square. Choose the best direction "
                "right now. Staying alive matters more than eating.")
RULE = ("Correct when no ghost is {d} or fewer steps away in this direction and this direction has the fewest steps to a "
        "pellet among the directions without a ghost that close. If every direction has a ghost {d} or fewer steps away, "
        "correct when this direction has the most steps to the nearest ghost.").format(d=DANGER)
CRITERIA_RULE = {move: f"Move {move}. {RULE}" for move in MOVES}
CRITERIA_JUDGEMENT = {move: f"Move one square {move}." for move in MOVES}
CRITERIA = {"rule": CRITERIA_RULE, "judgement": CRITERIA_JUDGEMENT, "outcomes": None, "situation": None, "described": None, "distance": None}  # the last four: written per turn
LEFT_OF = {"up": "left", "left": "down", "down": "right", "right": "up"}


ORDER = ("keep going", "turn left", "turn right", "reverse")


def relative(move: str, heading: str | None) -> str:
    """The move as the player sees it, relative to the way Pac-Man is heading."""
    if heading is None:
        return f"go {move}"
    if move == heading:
        return "keep going"
    if move == REVERSE[heading]:
        return "reverse"
    return "turn left" if move == LEFT_OF[heading] else "turn right"


def outcome(pellet: int | None, ghost: int | None) -> str:
    """What lies this way, in words, danger first (the "outcomes" version: code weighs the facts, the model chooses)."""
    food = ("eats a pellet right away" if pellet == 1 else "pellets a few steps along" if pellet is not None and pellet <= 4
            else "pellets further away" if pellet is not None else "nothing left to eat that way")
    if ghost is not None and ghost <= 2:
        return "deadly, you run into a ghost and lose a life."
    if ghost is not None and ghost <= 4:
        return f"risky, a ghost is close this way; {food}."
    return f"safe, no ghost close this way; {food}."


def situation(pellet: int | None, ghost: int | None) -> str:
    """What lies this way, in words, without saying whether that is good or bad."""
    danger = ("a ghost is right there" if ghost is not None and ghost <= 2 else
              "a ghost is a few steps away" if ghost is not None and ghost <= 4 else
              "a ghost is further along" if ghost is not None and ghost <= 8 else "no ghost this way")
    food = ("a pellet right there" if pellet == 1 else "pellets a few steps along" if pellet is not None and pellet <= 4
            else "pellets further on" if pellet is not None else "no pellets this way")
    return f"{danger}; {food}."


def open_square(square: tuple[int, int]) -> bool:
    row, col = square
    return 0 <= row < len(MAZE) and 0 <= col < len(MAZE[0]) and MAZE[row][col] != "#"


def neighbours(square: tuple[int, int]) -> dict[str, tuple[int, int]]:
    row, col = square
    return {move: (row + dr, col + dc) for move, (dr, dc) in MOVES.items() if open_square((row + dr, col + dc))}


def distances(source: tuple[int, int], blocked: tuple[int, int] | None = None) -> dict[tuple[int, int], int]:
    """Steps from source to every reachable square, not passing through `blocked`."""
    seen, queue = {source: 0}, deque([source])
    while queue:
        square = queue.popleft()
        for nxt in neighbours(square).values():
            if nxt not in seen and nxt != blocked:
                seen[nxt] = seen[square] + 1
                queue.append(nxt)
    return seen


class PacMan:
    name = "pacman"

    def __init__(self, criteria: str = "rule") -> None:
        self.version, self.criteria = criteria, CRITERIA[criteria]

    def reset(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.pellets = {(r, c) for r, line in enumerate(MAZE) for c, ch in enumerate(line) if ch == "."} - {START}
        self.total = len(self.pellets)
        self.lives, self.eaten = 3, 0
        self.place()

    def place(self) -> None:
        self.pac, self.pac_heading, self.named = START, None, {}
        self.previous_ghost_distance: list[int] | None = None
        self.ghosts = list(GHOST_STARTS)
        self.heading = [None, None]

    @property
    def done(self) -> bool:
        return self.lives == 0 or not self.pellets

    def leaving(self, move: str) -> dict[str, int | None]:
        """Steps to the nearest pellet and to the nearest ghost when Pac-Man's first step is `move`."""
        first = neighbours(self.pac)[move]
        reach = distances(first, blocked=self.pac)
        reach[first] = 0
        pellet = min((reach[p] + 1 for p in self.pellets if p in reach), default=None)
        ghost = min((reach[g] + 1 for g in self.ghosts if g in reach), default=None)
        return {"steps_to_nearest_pellet": pellet, "steps_to_nearest_ghost": ghost}

    def straight_pellets(self, move: str) -> int:
        """Pellets in a straight line that way, up to the next wall."""
        dr, dc = MOVES[move]
        count, square = 0, (self.pac[0] + dr, self.pac[1] + dc)
        while open_square(square):
            count += square in self.pellets
            square = (square[0] + dr, square[1] + dc)
        return count

    def describe_way(self, name: str, move: str) -> str:
        """Where the danger and the food are that way, in words (the "described" version)."""
        first = neighbours(self.pac)[move]
        reach = distances(first, blocked=self.pac)
        reach[first] = 0
        ways = [(reach[g] + 1, i) for i, g in enumerate(self.ghosts) if g in reach]
        if not ways:
            danger = "no ghost can be reached that way"
        else:
            steps, index = min(ways)
            closeness = "right next to you" if steps <= 1 else "very close" if steps <= 3 else "close" if steps <= 5 else "far"
            before = self.previous_ghost_distance[index] if self.previous_ghost_distance else None
            now = distances(self.ghosts[index]).get(self.pac)
            coming = ("coming towards you" if before is not None and now is not None and now < before else
                      "moving away from you" if before is not None and now is not None and now > before else None)
            danger = (f"the nearest ghost that way is {closeness} ({steps} {'step' if steps == 1 else 'steps'})"
                      + (f", {coming}" if coming else ""))
        straight = self.straight_pellets(move)
        pellet = self.leaving(move)["steps_to_nearest_pellet"]
        food = (f"{straight} {'pellet' if straight == 1 else 'pellets'} in a straight line ahead" if straight else
                f"no pellets straight ahead; the nearest pellet that way is {pellet} steps away" if pellet else "no pellets that way")
        back = "back the way you came; " if name == "reverse" else ""
        return f"{back}{food}; {danger}."

    def describe_distance(self, name: str, move: str) -> str:
        """The nearest ghost's distance now and after this move, plus the food that way (the "distance" version)."""
        now = min(distances(self.pac).get(g, 999) for g in self.ghosts)
        after = min(distances(neighbours(self.pac)[move]).get(g, 999) for g in self.ghosts)
        change = "closer" if after < now else "further away" if after > now else "the same"
        straight = self.straight_pellets(move)
        pellet = self.leaving(move)["steps_to_nearest_pellet"]
        food = (f"{straight} {'pellet' if straight == 1 else 'pellets'} ahead" if straight else
                f"nearest pellet {pellet} steps away" if pellet else "no pellets")
        back = "back the way you came; " if name == "reverse" else ""
        return f"{back}nearest ghost goes from {now} to {after} steps away ({change}); {food}."

    def render(self):
        """A drawn view of the maze for videos: Pac-Man chomping towards where it is heading, ghosts looking at it."""
        from PIL import Image, ImageDraw
        cell = 32
        image = Image.new("RGB", (len(MAZE[0]) * cell, len(MAZE) * cell), (5, 5, 20))
        draw = ImageDraw.Draw(image)
        for r, line in enumerate(MAZE):
            for c, ch in enumerate(line):
                if ch == "#":
                    draw.rounded_rectangle((c * cell + 3, r * cell + 3, c * cell + cell - 3, r * cell + cell - 3), 5,
                                           outline=(60, 90, 255), width=3)
        for r, c in self.pellets:
            draw.ellipse((c * cell + 13, r * cell + 13, c * cell + 19, r * cell + 19), fill=(250, 220, 180))
        pr, pc = self.pac
        for (r, c), colour in zip(self.ghosts, ((240, 50, 50), (250, 150, 220))):
            x, y = c * cell + 3, r * cell + 3
            size = cell - 6
            draw.pieslice((x, y, x + size, y + size), 180, 360, fill=colour)  # round head
            draw.rectangle((x, y + size // 2, x + size, y + size - 5), fill=colour)
            for k in range(3):  # wavy hem
                draw.polygon([(x + k * size / 3, y + size - 5), (x + (k + 0.5) * size / 3, y + size), (x + (k + 1) * size / 3, y + size - 5)],
                             fill=colour)
            look = (max(-1, min(1, pc - c)), max(-1, min(1, pr - r)))  # eyes look towards Pac-Man
            for ex in (x + 6, x + size - 14):
                draw.ellipse((ex, y + 7, ex + 8, y + 17), fill=(255, 255, 255))
                draw.ellipse((ex + 2 + 2 * look[0], y + 10 + 2 * look[1], ex + 6 + 2 * look[0], y + 14 + 2 * look[1]), fill=(30, 60, 200))
        facing = {"right": 0, "down": 90, "left": 180, "up": 270}[self.pac_heading or "left"]
        gape = 35 if (pr + pc) % 2 == 0 else 8  # chomp: every move changes row + column parity, so the mouth opens and closes
        draw.pieslice((pc * cell + 3, pr * cell + 3, pc * cell + cell - 3, pr * cell + cell - 3),
                      facing + gape, facing - gape + 360, fill=(255, 230, 0))
        return image

    def picture(self) -> list[str]:
        rows = [list(line.replace(".", " ")) for line in MAZE]
        for r, c in self.pellets:
            rows[r][c] = "."
        for r, c in self.ghosts:
            rows[r][c] = "G"
        rows[self.pac[0]][self.pac[1]] = "P"
        return ["".join(row) for row in rows]

    def observe(self) -> tuple[dict, dict]:
        legal = sorted(neighbours(self.pac))
        state = {"maze": self.picture(),
                 "legend": "P = you (Pac-Man), G = ghost, . = pellet, # = wall",
                 "you": {"row": self.pac[0], "column": self.pac[1], "lives": self.lives,
                         "pellets_eaten": self.eaten, "pellets_left": len(self.pellets)},
                 "directions": {move: self.leaving(move) for move in legal},
                 "note": "steps are counted along the corridors; null means none reachable that way"}
        if self.criteria is None:  # options named relative to the heading ("keep going", ...), mapped back in step()
            named = {relative(move, self.pac_heading): move for move in legal}
            self.named = named
            order = sorted(named, key=lambda name: (ORDER.index(name) if name in ORDER else len(ORDER), name))
            if self.version in ("described", "distance"):
                describe_move = self.describe_way if self.version == "described" else self.describe_distance
                criteria = {name: describe_move(name, named[name]) for name in order}
            else:
                describe = outcome if self.version == "outcomes" else situation
                criteria = {name: describe(state["directions"][named[name]]["steps_to_nearest_pellet"],
                                           state["directions"][named[name]]["steps_to_nearest_ghost"]) for name in order}
            # the options carry the facts, so the state stays short (no maze, no coordinates)
            state = {"lives": self.lives, "pellets_eaten": self.eaten, "pellets_left": len(self.pellets)}
        else:
            criteria = {move: self.criteria[move] for move in legal}
        return state, {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}

    def ghost_move(self, index: int) -> None:
        square, heading = self.ghosts[index], self.heading[index]
        options = {m: s for m, s in neighbours(square).items() if heading is None or m != REVERSE[heading] or len(neighbours(square)) == 1}
        chase = 0.9 if index == 0 else 0.7  # tuned so the rule bot clears the maze in about half the episodes
        if self.rng.random() < chase:
            reach = distances(self.pac)
            move = min(options, key=lambda m: (reach.get(options[m], 999), m))
        else:
            move = self.rng.choice(sorted(options))
        self.ghosts[index], self.heading[index] = options[move], move

    def step(self, move: str) -> None:
        move = self.named.get(move, move) if self.criteria is None else move
        before = self.pac
        self.pac, self.pac_heading = neighbours(self.pac)[move], move
        caught = self.pac in self.ghosts
        if not caught:
            if self.pac in self.pellets:
                self.pellets.remove(self.pac)
                self.eaten += 1
            old = list(self.ghosts)
            self.previous_ghost_distance = [distances(g).get(self.pac, 999) for g in self.ghosts]
            for index in range(len(self.ghosts)):
                self.ghost_move(index)
            # caught on the same square, or passing through each other
            caught = self.pac in self.ghosts or any(g == before and o == self.pac for g, o in zip(self.ghosts, old))
        if caught:
            self.lives -= 1
            self.place()

    def result(self) -> dict[str, float]:
        return {"score": self.eaten, "pellets_eaten": self.eaten, "pellets_total": self.total,
                "cleared": float(not self.pellets), "lives_left": self.lives}


def rule_player(state: dict, question: dict) -> str:
    """The hand-coded bar to beat: exactly the rule the options spell out."""
    directions = state["directions"]
    far = lambda m: directions[m]["steps_to_nearest_ghost"] is None or directions[m]["steps_to_nearest_ghost"] > DANGER  # noqa: E731
    safe = [m for m in question["criteria"] if far(m)]
    if safe:
        return min(safe, key=lambda m: (directions[m]["steps_to_nearest_pellet"] or 999, m))
    return max(question["criteria"], key=lambda m: (directions[m]["steps_to_nearest_ghost"] or 999, m))
