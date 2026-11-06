"""AlphaZero training loop: self-play -> replay buffer -> train -> evaluate, repeated.

Example (small run, a few minutes on a laptop CPU):
    python -m connect4.train --net mlp --iterations 10 --out runs/mlp_test
"""
import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from connect4.evaluate import DEFAULT_OPENING_MOVES, play_pairing
from connect4.mcts import MCTS
from connect4.networks import NETWORKS, NetworkEvaluator, alphazero_loss
from connect4.parallel import make_pool
from connect4.selfplay import mirror, self_play_game

LOG_FIELDS = ["iteration", "buffer_size", "policy_loss", "value_loss", "selfplay_time",
              "score_vs_greedy", "score_vs_pure_mcts", "time"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--net", choices=NETWORKS, default="mlp")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--games-per-iteration", type=int, default=50)
    p.add_argument("--simulations", type=int, default=50, help="MCTS simulations per move")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--dirichlet-alpha", type=float, default=1.0)
    p.add_argument("--temperature-moves", type=int, default=10)
    p.add_argument("--buffer-size", type=int, default=50_000, help="max samples kept (sliding window)")
    p.add_argument("--fresh-buffer", action="store_true",
                   help="empty the buffer before each self-play phase, so training only uses the newest games")
    p.add_argument("--train-steps", type=int, default=200, help="gradient steps per iteration")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=5, help="evaluate every n iterations")
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4, help="parallel processes for self-play and evaluation")
    p.add_argument("--train-threads", type=int, default=4, help="PyTorch threads for the gradient steps")
    p.add_argument("--out", type=Path, default=Path("runs/default"))
    return p.parse_args(argv)


def train_step(model, optimizer, buffer, batch_size, rng):
    idx = rng.integers(len(buffer), size=batch_size)
    states, pis, zs = zip(*(buffer[i] for i in idx))
    x = torch.from_numpy(np.stack(states))
    target_pi = torch.from_numpy(np.stack(pis)).float()
    target_z = torch.tensor(zs, dtype=torch.float32)
    logits, value = model(x)
    loss, policy_loss, value_loss = alphazero_loss(logits, value, target_pi, target_z)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return policy_loss.item(), value_loss.item()


def export_weights(model):
    """Model weights as NumPy arrays, which are sent to workers by value.

    Tensors would be shared through one file descriptor each on Linux, and with dozens of
    tensors per game that exceeds the limit on open files.
    """
    return {name: tensor.numpy() for name, tensor in model.state_dict().items()}


def self_play_worker(task):
    """Worker: play one game on its own copy of the network. The game's seed fixes all its randomness."""
    args, weights, seed = task
    rng = np.random.default_rng(seed)
    model = NETWORKS[args.net]()
    model.load_state_dict({name: torch.from_numpy(w) for name, w in weights.items()})
    mcts = MCTS(NetworkEvaluator(model), args.simulations, args.c_puct,
                dirichlet_alpha=args.dirichlet_alpha, seed=rng.integers(2**32))
    return self_play_game(mcts, rng, args.temperature_moves)


def evaluate(pool, checkpoint, args, seed):
    """Score of network + MCTS (no noise, greedy moves) against the baselines, with random openings."""
    agent = f"{args.net}:{checkpoint}"
    results = {}
    for i, (name, opponent) in enumerate([("greedy", "greedy"), ("pure_mcts", f"pure:{args.simulations}")]):
        s = play_pairing(pool, agent, opponent, args.eval_games, DEFAULT_OPENING_MOVES,
                         args.simulations, args.c_puct, seed + 100_000 * i)
        results[f"score_vs_{name}"] = (s["wins"] + 0.5 * s["draws"]) / args.eval_games
    return results


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.train_threads)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    model = NETWORKS[args.net]()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buffer = deque(maxlen=args.buffer_size)
    log = []

    with make_pool(args.workers) as pool:
        for it in range(1, args.iterations + 1):
            start = time.time()

            # 1. Self-play with the current network, one independently seeded game per task
            if args.fresh_buffer:
                buffer.clear()
            weights = export_weights(model)
            tasks = [(args, weights, seed) for seed in rng.integers(2**32, size=args.games_per_iteration)]
            for samples in pool.map(self_play_worker, tasks, chunksize=1):
                buffer.extend(samples)
                buffer.extend(mirror(sample) for sample in samples)
            selfplay_time = time.time() - start

            # 2. Train on random mini-batches from the buffer
            model.train()
            losses = [train_step(model, optimizer, buffer, args.batch_size, rng) for _ in range(args.train_steps)]
            policy_loss, value_loss = np.mean(losses, axis=0)

            row = {"iteration": it, "buffer_size": len(buffer), "policy_loss": policy_loss,
                   "value_loss": value_loss, "selfplay_time": selfplay_time}

            # 3. Save a checkpoint and evaluate it
            if it % args.eval_every == 0 or it == args.iterations:
                checkpoint = args.out / f"model_{it:03d}.pt"
                torch.save(model.state_dict(), checkpoint)
                row.update(evaluate(pool, checkpoint, args, int(rng.integers(2**32))))
            row["time"] = time.time() - start
            log.append(row)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)
            # rewritten every iteration so a crashed or interrupted run keeps its log
            with open(args.out / "log.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
                writer.writeheader()
                writer.writerows(log)
    return model, log


if __name__ == "__main__":
    train(parse_args())
