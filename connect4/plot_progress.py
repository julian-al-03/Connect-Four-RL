"""Plot how each checkpoint scores against the checkpoint before it, averaged over seeds.

Each group is LABEL=GLOB, where GLOB matches run folders with consecutive_*.csv files written
by connect4.evaluate (see the README). Quote it so the shell does not expand it:
    python -m connect4.plot_progress "mlp=runs/main/mlp_seed?" "cnn=runs/main/cnn_seed?" --out figures/main

A score above 0.5 means the later checkpoint beat the earlier one. Lines show the mean over
seeds, shaded bands the range from lowest to highest seed.
"""
import argparse
import csv
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from connect4.plot import COLORS, plot_curve, save, style_axis, style_score_axis, use_integer_iterations


def load_run(folder):
    """Returns {later iteration: score} for one run."""
    scores = {}
    for path in glob.glob(str(Path(folder) / "consecutive_*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                iteration = int(re.search(r"model_(\d+)\.pt", row["agent_a"]).group(1))
                scores[iteration] = float(row["score"])
    return scores


def load_group(pattern):
    """Returns (iterations, scores of shape (seeds, iterations)), using iterations all seeds have."""
    runs = [load_run(folder) for folder in sorted(glob.glob(pattern))]
    if not runs or not all(runs):
        raise SystemExit(f"no consecutive_*.csv files for every run matching {pattern}")
    iterations = sorted(set.intersection(*(set(run) for run in runs)))
    return np.array(iterations), np.array([[run[i] for i in iterations] for run in runs])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("groups", nargs="+", metavar="LABEL=GLOB")
    p.add_argument("--out", type=Path, default=Path("figures"))
    args = p.parse_args(argv)
    if len(args.groups) > len(COLORS):
        p.error(f"at most {len(COLORS)} groups per figure")

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for spec, color in zip(args.groups, COLORS):
        label, pattern = spec.split("=", 1)
        iterations, scores = load_group(pattern)
        print(f"{label}: {scores.shape[0]} seeds, {scores.shape[1]} pairings")
        plot_curve(ax, iterations, scores, color, label, markers=True)
    style_axis(ax, "Score against the previous checkpoint", "Training iteration", "Score (win = 1, draw = ½)")
    style_score_axis(ax)
    use_integer_iterations([ax])
    args.out.mkdir(parents=True, exist_ok=True)
    save(fig, [ax], args.out / "consecutive_progress.png")


if __name__ == "__main__":
    main()
