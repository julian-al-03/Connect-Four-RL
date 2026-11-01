import csv

import numpy as np
import pytest

from connect4.plot import COLUMNS, load_group, main


def write_log(folder, rows):
    folder.mkdir()
    with open(folder / "log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def make_rows(n):
    return [{"iteration": i, "policy_loss": 1.9 - i * 0.01, "value_loss": 0.4, "time": 60.0,
             "score_vs_greedy": 0.8 if i % 5 == 0 else "", "score_vs_pure_mcts": 0.6 if i % 5 == 0 else ""}
            for i in range(1, n + 1)]


def test_load_group_uses_iterations_reached_by_all_seeds(tmp_path):
    write_log(tmp_path / "net_seed0", make_rows(10))
    write_log(tmp_path / "net_seed1", make_rows(6))  # still running
    group = load_group(str(tmp_path / "net_seed*"))
    assert group["iteration"].shape == (2, 6)
    assert np.isnan(group["score_vs_greedy"][0, 0]) and group["score_vs_greedy"][0, 4] == 0.8


def test_missing_runs_give_a_clear_error(tmp_path):
    with pytest.raises(SystemExit):
        load_group(str(tmp_path / "nothing*"))


def test_main_writes_all_figures(tmp_path):
    write_log(tmp_path / "a_seed0", make_rows(10))
    write_log(tmp_path / "b_seed0", make_rows(3))  # no evaluation yet
    out = tmp_path / "figures"
    main([f"a={tmp_path / 'a_seed*'}", f"b={tmp_path / 'b_seed*'}", "--out", str(out)])
    assert {p.name for p in out.iterdir()} == {"scores.png", "losses.png", "score_by_time.png"}
