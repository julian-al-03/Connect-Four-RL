import csv

from connect4.train import parse_args, train


def test_training_smoke(tmp_path):
    args = parse_args(["--iterations", "2", "--games-per-iteration", "2", "--simulations", "5",
                       "--train-steps", "2", "--batch-size", "8", "--eval-every", "2",
                       "--eval-games", "2", "--out", str(tmp_path)])
    _, log = train(args)
    assert len(log) == 2 and "score_vs_greedy" in log[-1]
    assert (tmp_path / "model_002.pt").exists()
    with open(tmp_path / "log.csv") as f:
        assert len(list(csv.DictReader(f))) == 2
