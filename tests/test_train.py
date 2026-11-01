import csv

import pytest

from connect4.train import parse_args, train


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
