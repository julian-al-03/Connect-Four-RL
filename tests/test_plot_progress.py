import csv

import numpy as np
import pytest

from connect4.plot_progress import load_group, main


def write_score(folder, later, earlier, score):
    folder.mkdir(exist_ok=True)
    with open(folder / f"consecutive_{later}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["agent_a", "agent_b", "score"])
        writer.writeheader()
        writer.writerow({"agent_a": f"mlp:{folder}/model_{later:03d}.pt",
                         "agent_b": f"mlp:{folder}/model_{earlier:03d}.pt", "score": score})


def test_load_group_uses_iterations_all_seeds_have(tmp_path):
    write_score(tmp_path / "a_seed0", 20, 10, 0.6)
    write_score(tmp_path / "a_seed0", 30, 20, 0.7)
    write_score(tmp_path / "a_seed1", 20, 10, 0.5)  # still running
    iterations, scores = load_group(str(tmp_path / "a_seed?"))
    assert iterations.tolist() == [20]
    assert np.allclose(scores, [[0.6], [0.5]])


def test_missing_results_give_a_clear_error(tmp_path):
    (tmp_path / "a_seed0").mkdir()
    with pytest.raises(SystemExit):
        load_group(str(tmp_path / "a_seed?"))


def test_main_writes_the_figure(tmp_path):
    write_score(tmp_path / "a_seed0", 20, 10, 0.6)
    out = tmp_path / "figures"
    main([f"a={tmp_path / 'a_seed?'}", "--out", str(out)])
    assert (out / "consecutive_progress.png").exists()
