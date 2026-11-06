import csv

import numpy as np
import pytest

from connect4.networks import NETWORKS
from connect4.train import export_weights, parse_args, self_play_worker, train


@pytest.mark.parametrize("net", ["mlp", "cnn"])
def test_training_smoke(tmp_path, net):
    args = parse_args(["--net", net, "--iterations", "2", "--games-per-iteration", "2", "--simulations", "5",
                       "--train-steps", "2", "--batch-size", "8", "--eval-every", "2",
                       "--eval-games", "2", "--out", str(tmp_path)])
    _, log = train(args)
    assert len(log) == 2 and "score_vs_greedy" in log[-1]
    assert (tmp_path / "model_002.pt").exists()
    with open(tmp_path / "log.csv") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_results_do_not_depend_on_worker_count(tmp_path):
    def run(workers):
        args = parse_args(["--iterations", "1", "--games-per-iteration", "4", "--simulations", "5",
                           "--train-steps", "2", "--batch-size", "8", "--eval-games", "4",
                           "--workers", str(workers), "--out", str(tmp_path / str(workers))])
        return train(args)[1]

    one, three = run(1), run(3)
    for key in ["buffer_size", "policy_loss", "value_loss", "score_vs_greedy", "score_vs_pure_mcts"]:
        assert one[0][key] == three[0][key]


@pytest.mark.parametrize("net", ["mlp", "cnn"])
def test_workers_receive_plain_arrays_and_can_play(net):
    weights = export_weights(NETWORKS[net]())
    assert all(type(w) is np.ndarray for w in weights.values())  # no tensors: see export_weights
    args = parse_args(["--net", net, "--simulations", "5"])
    samples = self_play_worker((args, weights, 0))
    assert len(samples) >= 7


def test_fresh_buffer_keeps_only_the_newest_games(tmp_path):
    def run(*extra):
        args = parse_args(["--iterations", "2", "--games-per-iteration", "3", "--simulations", "5",
                           "--train-steps", "2", "--batch-size", "8", "--eval-every", "5",
                           "--out", str(tmp_path / str(len(extra))), *extra])
        return train(args)[1]

    sliding, fresh = run(), run("--fresh-buffer")
    # Same seeds, so both runs play the same games; the fresh buffer drops iteration 1's samples
    assert fresh[0]["buffer_size"] == sliding[0]["buffer_size"]
    assert fresh[1]["buffer_size"] == sliding[1]["buffer_size"] - sliding[0]["buffer_size"]
