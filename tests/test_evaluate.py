import csv

import numpy as np
import pytest
import torch

from connect4 import arena
from connect4.arena import play_game, play_match, random_opening
from connect4.evaluate import evaluate, parse_args, wilson_interval
from connect4.game import ConnectFour
from connect4.networks import MLPNet, PolicyAgent, ResNet


def first_legal(state):
    return int(np.flatnonzero(state.legal_moves())[0])


def test_random_opening_is_legal_and_reproducible():
    opening = random_opening(6, np.random.default_rng(0))
    state = ConnectFour()
    for col in opening:
        state = state.play(col)  # raises on an illegal move
    assert state.num_moves == 6 and not state.is_terminal()
    assert opening == random_opening(6, np.random.default_rng(0))
    with pytest.raises(ValueError):
        random_opening(7, np.random.default_rng(0))


def test_play_game_starts_from_opening():
    seen = []

    def spy(state):
        seen.append(state.num_moves)
        return first_legal(state)

    play_game(spy, spy, opening=[3, 3])
    assert seen[0] == 2


def test_each_opening_is_played_with_both_colours(monkeypatch):
    calls = []

    def fake_play_game(first, second, opening=()):
        calls.append((first, second, tuple(opening)))
        return 0

    monkeypatch.setattr(arena, "play_game", fake_play_game)
    play_match("a", "b", 20, opening_moves=4, seed=0)
    assert all(calls[g][:2] == ("a", "b") and calls[g + 1][:2] == ("b", "a") for g in range(0, 20, 2))
    assert all(calls[g][2] == calls[g + 1][2] for g in range(0, 20, 2))
    assert len({call[2] for call in calls}) == 10


def test_opening_results_are_from_agent_a_view():
    # first_legal always plays the lowest free column; with no opening the first player wins
    s = play_match(first_legal, first_legal, 4)
    assert s["wins"] == s["wins_as_first"] == 2 and s["losses"] == 2


def test_wilson_interval():
    low, high = wilson_interval(0.5, 200)
    assert low == pytest.approx(0.431, abs=0.001) and high == pytest.approx(0.569, abs=0.001)
    low, high = wilson_interval(1.0, 20)  # stays informative at a perfect score
    assert 0.8 < low < 1 and high == pytest.approx(1.0)


@pytest.mark.parametrize("make_net", [MLPNet, ResNet])
def test_policy_agent_plays_legal_moves(make_net):
    state = ConnectFour()
    for _ in range(6):
        state = state.play(0)
    assert PolicyAgent(make_net())(state) != 0


def test_evaluate_end_to_end(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save(MLPNet().state_dict(), checkpoint)
    out = tmp_path / "results.csv"
    args = parse_args(["--agents", f"mlp:{checkpoint}", f"mlp:{checkpoint}:0", "--opponents", "random", "pure:5",
                       "--games", "4", "--simulations", "5", "--out", str(out)])
    assert len(evaluate(args)) == 4
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and all(int(r["wins"]) + int(r["draws"]) + int(r["losses"]) == 4 for r in rows)


def test_odd_number_of_games_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--agents", "random", "greedy", "--games", "3"])
