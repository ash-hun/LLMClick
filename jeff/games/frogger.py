"""A small Frogger, deterministic for a given seed, played one decision per turn.

The board is 9 squares wide. Row 0 is the start, rows 1-3 are road (cars), row 4 is a safe median, rows 5-7 are river
(logs), row 8 is the goal. Each turn the frog moves one square (or stays), then cars and logs move. A car on the frog's
square kills it; on a river row the frog must stand on a log (which carries it along) or it drowns, and a log carrying
it off the board also drowns it. Reaching row 8 scores a crossing and the frog starts again at the bottom. Three lives.

The state is the same for both question versions: the board as text, plus what will be on each square the frog could
move to once cars and logs have moved (the fact the rule version refers to, like Doom's bearing)."""

import copy
import random

WIDTH, GOAL = 9, 8
ROAD, RIVER = (1, 2, 3), (5, 6, 7)
MOVES = {"up": (1, 0), "down": (-1, 0), "left": (0, -1), "right": (0, 1), "stay": (0, 0)}

INSTRUCTIONS = ("You are the frog in a game of Frogger. The goal is the top row. Each turn you move one square up, down, "
                "left or right, or stay where you are; then the cars and logs move. On a road row a car on your square "
                "kills you. On a river row you must be on a log, or you drown. Choose the best move right now.")
# RULE: each option says when it is the right move, in terms of the facts in the state (like the Doom harness's rule).
CRITERIA_RULE = {
    "up": "Move up. Correct whenever the square above is safe after cars and logs move. Moving up is how you score; "
          "do it the moment the square above is safe.",
    "stay": "Stay where you are. Correct only when the square above is not safe but your own square is safe.",
    "left": "Move left. Correct only when neither the square above nor your own square is safe, and the square to your "
            "left is safe.",
    "right": "Move right. Correct only when neither the square above nor your own square is safe, and the square to "
             "your right is safe.",
    "down": "Move down. Correct only when no other move is safe and the square below is safe.",
}
# JUDGEMENT: each option only says what the move does.
CRITERIA_JUDGEMENT = {
    "up": "Move one square up, towards the goal.",
    "stay": "Stay on your square this turn.",
    "left": "Move one square left.",
    "right": "Move one square right.",
    "down": "Move one square down, away from the goal.",
}
CRITERIA = {"rule": CRITERIA_RULE, "judgement": CRITERIA_JUDGEMENT, "outcomes": None, "situation": None, "described": None, "space": None, "space2": None, "zones": None}  # the last six: written per turn
LABEL = {"up": "Hop forward", "stay": "Wait", "left": "Step left", "right": "Step right", "down": "Hop back"}


def outcome(move: str, what: str) -> str:
    """What a move leads to, in words (the "outcomes" version: the model still chooses among the consequences)."""
    if what == "car":
        return f"{LABEL[move]}: you would be hit by a car and lose a life."
    if what in ("water", "log carrying you off the board"):
        return f"{LABEL[move]}: you would fall in the water and lose a life."
    if what == "goal":  # the same words as every other forward option, then the crossing: the 0.8B skips anything less familiar
        return f"{LABEL[move]}: safe, and one row closer to the goal: you reach it and score a crossing."
    progress = {"up": "safe, and one row closer to the goal", "down": "safe, but one row further from the goal"}
    return f"{LABEL[move]}: {progress.get(move, 'safe, but no progress')}."


SQUARE = {"car": "a car will be on that square", "water": "that square is open water",
          "log carrying you off the board": "onto a log that is about to leave the board", "empty road": "onto clear road",
          "log": "onto a log", "safe ground": "onto the grass", "goal": "onto the goal row"}


STAY = {"car": "a car will be on your square", "water": "you are in open water",
        "log carrying you off the board": "your log is about to leave the board", "empty road": "your square of road will be clear",
        "log": "you stay on your log", "safe ground": "you stay on the grass", "goal": "you stay on the goal row"}


def situation(move: str, what: str) -> str:
    """What is on the square a move leads to, in words, without saying whether that is good or bad."""
    if move == "stay":
        return f"Wait: {STAY[what]}."
    return f"{LABEL[move]}: {SQUARE[what]}."


class Lane:
    def __init__(self, cells: list[bool], direction: int, period: int) -> None:
        self.cells, self.direction, self.period = cells, direction, period  # cells: car or log on each square

    def shifted(self, turn: int) -> list[bool]:
        """The lane after the move that happens at the end of this turn."""
        if (turn + 1) % self.period:
            return self.cells
        return self.cells[-self.direction:] + self.cells[:-self.direction] if self.direction == 1 else self.cells[1:] + self.cells[:1]


class Frogger:
    name = "frogger"

    def __init__(self, criteria: str = "rule") -> None:
        self.version, self.criteria = criteria, CRITERIA[criteria]

    def reset(self, seed: int) -> None:
        rng = random.Random(seed)
        self.lanes: dict[int, Lane] = {}
        for row in ROAD:  # cars: short runs with gaps
            cells = [False] * WIDTH
            for start in rng.sample(range(0, WIDTH, 3), 2):
                for k in range(rng.choice((1, 2))):
                    cells[(start + k) % WIDTH] = True
            self.lanes[row] = Lane(cells, rng.choice((-1, 1)), rng.choice((1, 2)))
        for row in RIVER:  # logs: long runs with gaps, so some square is usually reachable
            cells = [False] * WIDTH
            start = rng.randrange(WIDTH)
            for k in range(rng.choice((3, 4, 5))):
                cells[(start + k) % WIDTH] = True
            self.lanes[row] = Lane(cells, rng.choice((-1, 1)), rng.choice((1, 2)))
        self.row, self.col, self.turn, self.lives, self.crossings, self.best_row = 0, WIDTH // 2, 0, 3, 0, 0
        self.just_crossed: int | None = None  # the column of a crossing made on the last turn (for videos)

    @property
    def done(self) -> bool:
        return self.lives == 0

    def kind(self, row: int) -> str:
        return "road" if row in ROAD else "river" if row in RIVER else "goal" if row == GOAL else "safe ground"

    def after(self, row: int, col: int) -> str:
        """What the frog would face on (row, col) once cars and logs have moved at the end of this turn."""
        if not (0 <= row <= GOAL and 0 <= col < WIDTH):
            return "off the board"
        if row in ROAD:
            return "car" if self.lanes[row].shifted(self.turn)[col] else "empty road"
        if row in RIVER:  # standing on a log moves the frog with it
            lane = self.lanes[row]
            if not lane.cells[col]:
                return "water"
            moved = (self.turn + 1) % lane.period == 0
            new = col + (lane.direction if moved else 0)
            return "log" if 0 <= new < WIDTH else "log carrying you off the board"
        return self.kind(row)

    def car_arrives(self, row: int, col: int, horizon: int = 5) -> int | None:
        """Turns until a car covers (row, col) after cars move: 1 = at the end of this turn; None = not within the horizon."""
        lane, cells = self.lanes[row], self.lanes[row].cells
        for turn in range(horizon):
            cells = Lane(cells, lane.direction, lane.period).shifted(self.turn + turn)
            if cells[col]:
                return turn + 1
        return None

    def describe_square(self, move: str) -> str:
        """Where the danger is and how far, in words, for the square a move leads to (the "described" version)."""
        dr, dc = MOVES[move]
        row, col = self.row + dr, self.col + dc
        progress = {"up": "one row closer to the goal", "down": "one row further from the goal"}.get(move, "no progress")
        label = "Wait" if move == "stay" else LABEL[move]
        if row == GOAL:
            return f"{label}: you reach the goal."
        if row in ROAD:
            lane = self.lanes[row]
            side = "left" if lane.direction == 1 else "right"
            when = self.car_arrives(row, col)
            square = "your square" if move == "stay" else "that square"
            traffic = (f"a car will be on {square} when the cars move" if when == 1 else
                       f"the next car reaches {square} in {when} turns, coming from your {side}" if when else
                       f"no car reaches {square} for at least 5 turns")
            where = "you stay on the road" if move == "stay" else f"onto road ({progress})"
            return f"{label}: {where}; {traffic}."
        if row in RIVER:
            lane = self.lanes[row]
            drift = "right" if lane.direction == 1 else "left"
            if not lane.cells[col]:
                logs = [c for c in range(WIDTH) if lane.cells[c]]
                nearest = min(logs, key=lambda c: abs(c - col)) if logs else None
                where = (f"the nearest log in that row is {abs(nearest - col)} squares to your {'right' if nearest > col else 'left'}"
                         if nearest is not None else "there is no log in that row")
                return f"{label}: into open water ({progress}); {where}, drifting {drift}."
            ahead, c = 0, col + lane.direction
            while 0 <= c < WIDTH and lane.cells[c]:
                ahead, c = ahead + 1, c + lane.direction
            edge = (WIDTH - 1 - col) if lane.direction == 1 else col
            return (f"{label}: {'you stay on your log' if move == 'stay' else f'onto a log ({progress})'}; the log drifts {drift} one square every "
                    f"{'turn' if lane.period == 1 else '2 turns'}; you are {edge} squares from the {drift} edge of the board.")
        return f"{label}: {'you stay on the grass' if move == 'stay' else f'onto the grass ({progress})'}."

    def free_moves(self) -> int:
        """How many of the five moves would land on a free square this turn."""
        return sum(self.safe(self.after(self.row + dr, self.col + dc)) for dr, dc in MOVES.values())

    def describe_space(self, move: str) -> str:
        """Is the square free, and how much room does it leave for the next move (the "space" version)."""
        dr, dc = MOVES[move]
        label = "Wait" if move == "stay" else LABEL[move]
        direction = {"up": "forward", "down": "back"}.get(move, "sideways" if move != "stay" else "in place")
        what = self.after(self.row + dr, self.col + dc)
        if what == "car":
            return f"{label}: blocked, a car will be on that square."
        if what in ("water", "log carrying you off the board"):
            return f"{label}: blocked, {'open water' if what == 'water' else 'the log carries you off the edge'}."
        if what == "goal":
            return f"{label}: free, reaches the goal."
        ahead = copy.deepcopy(self)
        ahead.step(move)
        room = ahead.free_moves()
        space = ("plenty of room" if room >= 4 else "some room" if room >= 2 else "boxed in" if room == 1 else "trapped")
        return f"{label}: free square, {direction}; {space} there ({room} of 5 next moves free)."

    def describe_space2(self, move: str) -> str:
        """Is the square free, and which of the next moves from there are free, by direction (the "space2" version)."""
        dr, dc = MOVES[move]
        label = "Wait" if move == "stay" else LABEL[move]
        progress = {"up": "one row closer to the goal", "down": "one row back"}.get(move, "no progress")
        what = self.after(self.row + dr, self.col + dc)
        if what == "car":
            return f"{label}: blocked, a car will be on that square."
        if what in ("water", "log carrying you off the board"):
            return f"{label}: blocked, {'open water' if what == 'water' else 'the log carries you off the edge'}."
        if what == "goal":
            return f"{label}: free, reaches the goal."
        ahead = copy.deepcopy(self)
        ahead.step(move)
        free = {m: ahead.safe(ahead.after(ahead.row + r, ahead.col + c)) for m, (r, c) in MOVES.items()}
        sideways = sum(free[m] for m in ("left", "right", "stay"))
        return (f"{label}: free, {progress}; next turn from there: forward {'free' if free['up'] else 'blocked'}, "
                f"{sideways} of 3 sideways or waiting moves free, back {'free' if free['down'] else 'blocked'}.")

    ZONE_INSTRUCTIONS = {
        "road": ("You are the frog in Frogger, crossing the road. Cars drive along each row. Get to the top of the screen "
                 "without being hit. Each turn you hop one square or wait; then the cars move. Choose the best move right now."),
        "river": ("You are the frog in Frogger, crossing the river. You can only stand on logs: water drowns you. The logs "
                  "drift sideways and can carry you off the edge of the screen. Get to the top without drowning. Each turn "
                  "you hop one square or wait; then the logs drift. Choose the best move right now."),
    }

    def zone(self) -> str:
        """The part of the crossing the frog is dealing with: the road until it reaches the median, then the river."""
        return "river" if self.row >= 4 else "road"

    def describe_zone(self, move: str) -> str:
        """The consequence of a move in words, with log details on the river (the "zones" version)."""
        dr, dc = MOVES[move]
        row, col = self.row + dr, self.col + dc
        what = self.after(row, col)
        label = "Wait" if move == "stay" else LABEL[move]
        progress = {"up": "one row closer to the goal", "down": "one row further from the goal"}.get(move, "no progress")
        if what == "car":
            return f"{label}: you would be hit by a car and lose a life."
        if what == "water":
            return f"{label}: you would land in the water and drown."
        if what == "log carrying you off the board":
            return f"{label}: the log would carry you off the edge and you would drown."
        if what == "goal":
            return f"{label}: you reach the goal and score a crossing."
        if what == "log":
            lane = self.lanes[row]
            drift = "right" if lane.direction == 1 else "left"
            moved = (self.turn + 1) % lane.period == 0
            new = col + (lane.direction if moved else 0)
            edge = (WIDTH - 1 - new) if lane.direction == 1 else new
            where = "you stay on your log" if move == "stay" else f"you land on a log, {progress}"
            return f"{label}: {where}; it drifts {drift} and you would be {edge} squares from the {drift} edge."
        ground = "the road" if what == "empty road" else "the grass"
        return f"{label}: {'you stay on ' + ground if move == 'stay' else 'you land safely on ' + ground + ', ' + progress}."

    def render(self):
        """A drawn view of the board for videos: road, river, grass, goal; cars with windows and wheels, logs, the frog."""
        from PIL import Image, ImageDraw
        cell = 48
        image = Image.new("RGB", (WIDTH * cell, (GOAL + 1) * cell))
        draw = ImageDraw.Draw(image)
        for row in range(GOAL + 1):
            y = (GOAL - row) * cell
            ground = (70, 70, 78) if row in ROAD else (40, 90, 170) if row in RIVER else (200, 170, 40) if row == GOAL else (60, 140, 60)
            draw.rectangle((0, y, WIDTH * cell, y + cell), fill=ground)
            if row in ROAD:  # lane markings
                for x in range(0, WIDTH * cell, 24):
                    draw.line((x, y + cell - 1, x + 12, y + cell - 1), fill=(150, 150, 150), width=2)
            if row in self.lanes:
                lane = self.lanes[row]
                for col, filled in enumerate(lane.cells):
                    if not filled:
                        continue
                    x = col * cell
                    if row in ROAD:
                        draw.rounded_rectangle((x + 4, y + 10, x + cell - 4, y + cell - 10), 7, fill=(210, 50, 50))
                        front = x + cell - 16 if lane.direction == 1 else x + 8
                        draw.rectangle((front, y + 15, front + 8, y + cell - 15), fill=(170, 220, 255))  # windscreen
                        for wx in (x + 10, x + cell - 16):
                            draw.rectangle((wx, y + 6, wx + 6, y + 10), fill=(20, 20, 20))
                            draw.rectangle((wx, y + cell - 10, wx + 6, y + cell - 6), fill=(20, 20, 20))
                    else:
                        draw.rectangle((x, y + 9, x + cell, y + cell - 9), fill=(130, 85, 40))
                        draw.line((x + 4, y + 20, x + cell - 4, y + 20), fill=(100, 62, 28), width=2)
                        draw.line((x + 4, y + 30, x + cell - 4, y + 30), fill=(100, 62, 28), width=2)
        def frog(col: int, row: int, faded: bool = False) -> None:
            x, y = col * cell, (GOAL - row) * cell
            body, dark = ((150, 240, 150), (60, 120, 60)) if faded else ((70, 200, 80), (20, 90, 30))
            for lx, ly in ((x + 6, y + 8), (x + cell - 16, y + 8), (x + 6, y + cell - 18), (x + cell - 16, y + cell - 18)):
                draw.ellipse((lx, ly, lx + 10, ly + 10), fill=dark)  # legs
            draw.ellipse((x + 10, y + 10, x + cell - 10, y + cell - 8), fill=body, outline=dark, width=2)
            for ex in (x + 14, x + cell - 24):
                draw.ellipse((ex, y + 8, ex + 10, y + 18), fill=(255, 255, 255))
                draw.ellipse((ex + 3, y + 10, ex + 7, y + 14), fill=(0, 0, 0))
        if self.just_crossed is not None:  # show the frog arriving on the goal row before it starts again
            frog(self.just_crossed, GOAL, faded=True)
        frog(self.col, self.row)
        return image

    def safe(self, what: str) -> bool:
        return what in ("empty road", "log", "safe ground", "goal")

    def board(self) -> list[str]:
        lines = []
        for row in range(GOAL, -1, -1):
            if row in self.lanes:
                lane = self.lanes[row]
                mark = "C" if row in ROAD else "L"
                squares = "".join(mark if cell else ("." if row in ROAD else "~") for cell in lane.cells)
                where = f"{self.kind(row)}, moving {'right' if lane.direction == 1 else 'left'} every {'turn' if lane.period == 1 else '2 turns'}"
            else:
                squares, where = "_" * WIDTH, self.kind(row)
            if row == self.row:
                squares = squares[:self.col] + "F" + squares[self.col + 1:]
            lines.append(f"row {row} ({where}): {squares}")
        return lines

    def observe(self) -> tuple[dict, dict]:
        around = {move: self.after(self.row + dr, self.col + dc) for move, (dr, dc) in MOVES.items()}
        state = {"board_top_to_bottom": self.board(),
                 "legend": "F = you (the frog), C = car, L = log, ~ = water, . = empty road, _ = safe ground; columns 0-8 left to right",
                 "you": {"row": self.row, "column": self.col, "lives": self.lives, "crossings": self.crossings},
                 "square_above_after_cars_and_logs_move": around["up"],
                 "your_square_after_cars_and_logs_move": around["stay"],
                 "square_left_after_cars_and_logs_move": around["left"],
                 "square_right_after_cars_and_logs_move": around["right"],
                 "square_below_after_cars_and_logs_move": around["down"]}
        legal = [move for move in MOVES if around[move] != "off the board"]
        if self.criteria is None:  # the options carry the facts, so the state stays short (no board, no coordinates)
            if self.version == "zones":
                criteria = {move: self.describe_zone(move) for move in legal}
                state = {"crossing": self.zone(), "rows_to_the_goal": GOAL - self.row, "lives": self.lives, "crossings": self.crossings}
                return state, {"type": "choice", "instructions": self.ZONE_INSTRUCTIONS[self.zone()], "criteria": criteria}
            if self.version in ("described", "space", "space2"):
                describe_move = {"described": self.describe_square, "space": self.describe_space}.get(self.version, self.describe_space2)
                criteria = {move: describe_move(move) for move in legal}
            else:
                describe = outcome if self.version == "outcomes" else situation
                criteria = {move: describe(move, around[move]) for move in legal}
            state = {"you_are_on": self.kind(self.row), "rows_to_the_goal": GOAL - self.row, "lives": self.lives,
                     "crossings": self.crossings}
        else:
            criteria = {move: self.criteria[move] for move in legal}
        return state, {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}

    def step(self, move: str) -> None:
        self.just_crossed = None
        dr, dc = MOVES[move]
        self.row, self.col = self.row + dr, self.col + dc
        fate = self.after(self.row, self.col)
        for row, lane in self.lanes.items():
            if row == self.row and row in RIVER and lane.cells[self.col] and (self.turn + 1) % lane.period == 0:
                self.col += lane.direction  # carried by the log
            lane.cells = lane.shifted(self.turn)
        self.turn += 1
        self.best_row = max(self.best_row, self.row)
        if not self.safe(fate):
            self.lives -= 1
            self.row, self.col = 0, WIDTH // 2
        elif self.row == GOAL:
            self.crossings += 1
            self.just_crossed = self.col
            self.row, self.col = 0, WIDTH // 2
            return

    def result(self) -> dict[str, float]:
        return {"score": self.crossings, "crossings": self.crossings, "furthest_row": self.best_row, "lives_left": self.lives}


def rule_player(state: dict, question: dict) -> str:
    """The hand-coded bar to beat: exactly the rule the options spell out."""
    safe = lambda key: state[key] in ("empty road", "log", "safe ground", "goal")  # noqa: E731
    if safe("square_above_after_cars_and_logs_move"):
        return "up"
    if safe("your_square_after_cars_and_logs_move"):
        return "stay"
    for move, key in (("left", "square_left_after_cars_and_logs_move"), ("right", "square_right_after_cars_and_logs_move"),
                      ("down", "square_below_after_cars_and_logs_move")):
        if move in question["criteria"] and safe(key):
            return move
    return "up"
