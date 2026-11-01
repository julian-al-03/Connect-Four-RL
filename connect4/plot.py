"""Plot learning curves from training logs, averaged over the seeds of each group.

Each group is LABEL=GLOB, where GLOB matches run folders that contain a log.csv. Quote it so
the shell does not expand it:
    python -m connect4.plot "mlp=runs/main/mlp_seed*" "cnn=runs/main/cnn_seed*" --out figures

Lines show the mean over seeds, shaded bands the range from lowest to highest seed. Only
iterations reached by every seed of a group are used, so unfinished runs can be plotted too.
"""
import argparse
import csv
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

COLUMNS = ["iteration", "policy_loss", "value_loss", "score_vs_greedy", "score_vs_pure_mcts", "time"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order: a group keeps its colour across figures
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def load_group(pattern):
    """Returns {column: array of shape (seeds, iterations)}; missing evaluations are NaN."""
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(Path(path) / "log.csv") as f:
            rows = list(csv.DictReader(f))
        runs.append({col: np.array([float(r[col]) if r[col] else np.nan for r in rows]) for col in COLUMNS})
    if not runs:
        raise SystemExit(f"no runs match {pattern}")
    n = min(len(run["iteration"]) for run in runs)
    return {col: np.stack([run[col][:n] for run in runs]) for col in COLUMNS}


def plot_curve(ax, x, ys, color, label, markers=False, end_label=False):
    """Mean line and min-max band of ys (seeds, points), skipping points without data."""
    keep = ~np.isnan(ys).any(axis=0)
    x, ys = x[keep], ys[:, keep]
    if not len(x):
        return
    ax.fill_between(x, ys.min(axis=0), ys.max(axis=0), color=color, alpha=0.15, linewidth=0)
    style = {"marker": "o", "markersize": 6, "markeredgecolor": "white", "markeredgewidth": 1.5} if markers else {}
    ax.plot(x, ys.mean(axis=0), color=color, linewidth=2, solid_capstyle="round", label=label, **style)
    if end_label:
        ax.annotate(label, (x[-1], ys.mean(axis=0)[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=TEXT)


def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=TEXT)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def style_score_axis(ax):
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color=MUTED, linewidth=1, zorder=1)
    ax.text(1.0, 0.5, "even", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8, color=MUTED)


def save(fig, axes, path):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, labelcolor=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def use_integer_iterations(axes):
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_scores(groups, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    panels = [("score_vs_pure_mcts", "Score against pure MCTS"), ("score_vs_greedy", "Score against greedy")]
    for ax, (column, title) in zip(axes, panels):
        for (label, group), color in zip(groups.items(), COLORS):
            plot_curve(ax, group["iteration"][0], group[column], color, label, markers=True)
        style_axis(ax, title, "Training iteration", "Score (win = 1, draw = ½)")
        style_score_axis(ax)
    axes[1].set_ylabel("")
    use_integer_iterations(axes)
    save(fig, axes, path)


def plot_losses(groups, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    panels = [("policy_loss", "Policy loss"), ("value_loss", "Value loss")]
    for ax, (column, title) in zip(axes, panels):
        for (label, group), color in zip(groups.items(), COLORS):
            plot_curve(ax, group["iteration"][0], group[column], color, label)
        style_axis(ax, title, "Training iteration", "Loss")
    use_integer_iterations(axes)
    save(fig, axes, path)


def plot_score_by_time(groups, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for (label, group), color in zip(groups.items(), COLORS):
        hours = np.cumsum(group["time"], axis=1).mean(axis=0) / 3600
        plot_curve(ax, hours, group["score_vs_pure_mcts"], color, label, markers=True, end_label=True)
    style_axis(ax, "Score against pure MCTS by training time", "Mean training time (hours)", "Score (win = 1, draw = ½)")
    style_score_axis(ax)
    ax.margins(x=0.1)  # room for the labels at the line ends
    save(fig, [ax], path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("groups", nargs="+", metavar="LABEL=GLOB")
    p.add_argument("--out", type=Path, default=Path("figures"))
    args = p.parse_args(argv)
    if len(args.groups) > len(COLORS):
        p.error(f"at most {len(COLORS)} groups per figure")

    groups = {}
    for spec in args.groups:
        label, pattern = spec.split("=", 1)
        groups[label] = load_group(pattern)
        print(f"{label}: {groups[label]['time'].shape[0]} seeds, {groups[label]['time'].shape[1]} iterations")

    args.out.mkdir(parents=True, exist_ok=True)
    plot_scores(groups, args.out / "scores.png")
    plot_losses(groups, args.out / "losses.png")
    plot_score_by_time(groups, args.out / "score_by_time.png")


if __name__ == "__main__":
    main()
