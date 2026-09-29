from jeff.games import common, frogger, pacman
from jeff.games.frogger import Frogger, Lane
from jeff.games.pacman import MAZE, PacMan, distances


def test_lane_shifts_in_its_direction_on_its_period() -> None:
    lane = Lane([True, False, False], direction=1, period=1)
    assert lane.shifted(0) == [False, True, False]
    assert Lane([True, False, False], direction=-1, period=1).shifted(0) == [False, False, True]
    assert Lane([True, False, False], direction=1, period=2).shifted(0) == [True, False, False]  # moves every 2 turns
    assert Lane([True, False, False], direction=1, period=2).shifted(1) == [False, True, False]


def test_frogger_car_kills_and_crossing_scores() -> None:
    game = Frogger()
    game.reset(1)
    game.lanes[1] = Lane([True] * 9, direction=1, period=1)  # a solid wall of cars
    game.step("up")
    assert game.lives == 2 and (game.row, game.col) == (0, 4)
    game.reset(1)
    game.row = 7
    game.lanes[7] = Lane([True] * 9, direction=1, period=2)  # frog on a log that does not move this turn
    game.step("up")
    assert game.crossings == 1 and (game.row, game.col) == (0, 4)


def test_frogger_water_drowns_and_log_carries() -> None:
    game = Frogger()
    game.reset(2)
    game.row, game.col = 4, 4
    game.lanes[5] = Lane([False] * 9, direction=1, period=1)
    game.step("up")
    assert game.lives == 2
    game.reset(2)
    game.row, game.col = 4, 4
    game.lanes[5] = Lane([True] * 9, direction=1, period=1)
    game.step("up")
    assert game.lives == 3 and (game.row, game.col) == (5, 5)


def test_frogger_state_names_the_facts_the_rule_uses() -> None:
    game = Frogger()
    game.reset(3)
    state, question = game.observe()
    assert "down" not in question["criteria"]  # off the board from the start row
    assert state["your_square_after_cars_and_logs_move"] == "safe ground"
    assert "Correct whenever" in question["criteria"]["up"]
    judged = Frogger("judgement")
    judged.reset(3)
    assert "Correct" not in judged.observe()[1]["criteria"]["up"]


def test_maze_is_rectangular_connected_and_has_no_dead_ends() -> None:
    assert len({len(line) for line in MAZE}) == 1
    open_squares = {(r, c) for r, line in enumerate(MAZE) for c, ch in enumerate(line) if ch != "#"}
    assert set(distances(pacman.START)) == open_squares
    assert all(len(pacman.neighbours(square)) >= 2 for square in open_squares)


def test_pacman_eats_pellets_and_loses_a_life_to_a_ghost() -> None:
    game = PacMan()
    game.reset(1)
    left = len(game.pellets)
    game.ghosts = [(1, 1), (1, 2)]  # far away
    game.step("left")
    assert game.eaten == 1 and len(game.pellets) == left - 1
    game.ghosts = [(game.pac[0], game.pac[1] - 1), (1, 1)]  # a ghost right next to Pac-Man
    game.step("left")
    assert game.lives == 2 and game.pac == pacman.START


def test_pacman_directions_report_steps() -> None:
    game = PacMan()
    game.reset(1)
    state, question = game.observe()
    assert set(question["criteria"]) == set(state["directions"]) == {"left", "right", "up"}
    assert state["directions"]["left"]["steps_to_nearest_pellet"] == 1


def test_runner_records_scores_and_rejects_illegal_moves() -> None:
    result = common.run(Frogger(), frogger.rule_player, [1, 2], max_steps=50)
    assert len(result["episodes"]) == 2 and result["decisions"] > 0
    try:
        common.run(PacMan(), lambda state, question: "fly", [1], max_steps=5)
    except ValueError as error:
        assert "not one of" in str(error)
    else:
        raise AssertionError("an illegal move was accepted")


def test_rule_bots_beat_random_play() -> None:
    seeds = list(range(1234, 1244))
    for game, bot, steps in ((Frogger(), frogger.rule_player, 300), (PacMan(), pacman.rule_player, 600)):
        rule = common.run(game, bot, seeds, steps)["mean_score"]
        rand = common.run(game, common.random_player(1234), seeds, steps)["mean_score"]
        assert rule > rand, (game.name, rule, rand)
