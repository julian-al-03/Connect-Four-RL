"""Play head-to-head matches between agents and report scores with confidence intervals.

Agents are given as strings:
    random, greedy      baselines
    pure:SIMS           MCTS with random rollouts
    NET:CHECKPOINT      network + MCTS with --simulations simulations, e.g. cnn:runs/a/model_020.pt
    NET:CHECKPOINT:SIMS the same with SIMS simulations; SIMS = 0 is the raw policy without search

Every pair of agents plays each other (round robin), unless --opponents is given: then each
of --agents plays each of --opponents. Example:
    python -m connect4.evaluate --agents cnn:runs/a/model_020.pt mlp:runs/b/model_020.pt \
        --opponents greedy pure:50 --games 200 --out runs/results.csv
"""
import argparse
import csv
import itertools
import math
import time

import torch

from connect4.agents import GreedyAgent, RandomAgent, pure_mcts_agent
from connect4.arena import play_match
from connect4.mcts import MCTS, MCTSAgent
from connect4.networks import NETWORKS, NetworkEvaluator, PolicyAgent
from connect4.parallel import make_pool

RESULT_FIELDS = ["agent_a", "agent_b", "games", "wins", "draws", "losses",
                 "score", "ci_low", "ci_high", "time"]
DEFAULT_OPENING_MOVES = 4
GAMES_PER_TASK = 10  # must be even: each opening is played with both colour assignments


def make_agent(spec, default_simulations, c_puct, seed):
    kind, *rest = spec.split(":")
    if kind == "random":
        return RandomAgent(seed)
    if kind == "greedy":
        return GreedyAgent(seed)
    if kind == "pure":
        return pure_mcts_agent(int(rest[0]), seed)
    if kind in NETWORKS:
        model = NETWORKS[kind]()
        model.load_state_dict(torch.load(rest[0]))
        simulations = int(rest[1]) if len(rest) > 1 else default_simulations
        if simulations == 0:
            return PolicyAgent(model)
        return MCTSAgent(MCTS(NetworkEvaluator(model), simulations, c_puct))
    raise ValueError(f"unknown agent: {spec}")


def wilson_interval(score, n, z=1.96):
    """95% Wilson confidence interval for a score in [0, 1] over n games.

    Unlike score +- z * standard error, it stays meaningful for scores near 0 or 1. Draws count
    as half a win and the games are treated as independent, so the interval is approximate.
    """
    denominator = 1 + z**2 / n
    center = (score + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(score * (1 - score) / n + z**2 / (4 * n**2)) / denominator
    return center - half_width, center + half_width


def _play_task(task):
    """Worker: build both agents and play part of a match."""
    spec_a, spec_b, n_games, opening_moves, simulations, c_puct, seed = task
    a = make_agent(spec_a, simulations, c_puct, seed)
    b = make_agent(spec_b, simulations, c_puct, seed + 1)
    return play_match(a, b, n_games, opening_moves, seed=seed + 2)


def play_pairing(pool, spec_a, spec_b, games, opening_moves, simulations, c_puct, seed):
    """Play a match in parallel. Tasks have fixed seeds, so the worker count does not change results."""
    sizes = [GAMES_PER_TASK] * (games // GAMES_PER_TASK)
    if games % GAMES_PER_TASK:
        sizes.append(games % GAMES_PER_TASK)
    tasks = [(spec_a, spec_b, n, opening_moves, simulations, c_puct, seed + 3 * i)
             for i, n in enumerate(sizes)]
    stats = {}
    for part in pool.map(_play_task, tasks, chunksize=1):
        for key, value in part.items():
            stats[key] = stats.get(key, 0) + value
    return stats


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", nargs="+", required=True)
    p.add_argument("--opponents", nargs="+", default=None)
    p.add_argument("--games", type=int, default=200, help="games per pairing (must be even)")
    p.add_argument("--opening-moves", type=int, default=DEFAULT_OPENING_MOVES, help="random moves at the start of each game")
    p.add_argument("--simulations", type=int, default=50)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4, help="parallel processes")
    p.add_argument("--out", default=None, help="CSV file for the results")
    args = p.parse_args(argv)
    if args.games % 2:
        p.error("--games must be even, so that each opening is played with both colour assignments")
    return args


def evaluate(args):
    if args.opponents:
        pairings = list(itertools.product(args.agents, args.opponents))
    else:
        pairings = list(itertools.combinations(args.agents, 2))
    print(f"{'A':<45} {'B':<45} {'W':>5} {'D':>5} {'L':>5} {'score A':>8} {'95% CI':>15}")
    results = []
    with make_pool(args.workers) as pool:
        for i, (spec_a, spec_b) in enumerate(pairings):
            start = time.time()
            s = play_pairing(pool, spec_a, spec_b, args.games, args.opening_moves,
                             args.simulations, args.c_puct, args.seed + 100_000 * i)
            score = (s["wins"] + 0.5 * s["draws"]) / args.games
            ci_low, ci_high = wilson_interval(score, args.games)
            results.append({"agent_a": spec_a, "agent_b": spec_b, "games": args.games,
                            "wins": s["wins"], "draws": s["draws"], "losses": s["losses"],
                            "score": score, "ci_low": ci_low, "ci_high": ci_high,
                            "time": time.time() - start})
            print(f"{spec_a:<45} {spec_b:<45} {s['wins']:>5} {s['draws']:>5} {s['losses']:>5} "
                  f"{score:>8.3f}   [{ci_low:.3f}, {ci_high:.3f}]", flush=True)
            if args.out:
                with open(args.out, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
                    writer.writeheader()
                    writer.writerows(results)
    return results


if __name__ == "__main__":
    evaluate(parse_args())
