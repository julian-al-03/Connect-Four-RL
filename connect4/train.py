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

from connect4.agents import GreedyAgent, pure_mcts_agent
from connect4.arena import play_match
from connect4.mcts import MCTS, MCTSAgent
from connect4.networks import MLPNet, NetworkEvaluator, alphazero_loss
from connect4.selfplay import mirror, self_play_game

NETWORKS = {"mlp": MLPNet}
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
    p.add_argument("--train-steps", type=int, default=200, help="gradient steps per iteration")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=5, help="evaluate every n iterations")
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
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


def evaluate(model, args):
    """Win rate of network + MCTS (no noise, greedy moves) against the baselines."""
    agent = MCTSAgent(MCTS(NetworkEvaluator(model), args.simulations, args.c_puct))
    results = {}
    for name, opponent in [("greedy", GreedyAgent(args.seed)),
                           ("pure_mcts", pure_mcts_agent(args.simulations, args.seed))]:
        s = play_match(agent, opponent, args.eval_games)
        results[f"score_vs_{name}"] = (s["wins"] + 0.5 * s["draws"]) / args.eval_games
    return results


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    model = NETWORKS[args.net]()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buffer = deque(maxlen=args.buffer_size)
    log = []

    for it in range(1, args.iterations + 1):
        start = time.time()

        # 1. Self-play with the current network
        mcts = MCTS(NetworkEvaluator(model), args.simulations, args.c_puct,
                    dirichlet_alpha=args.dirichlet_alpha, seed=rng.integers(2**32))
        for _ in range(args.games_per_iteration):
            samples = self_play_game(mcts, rng, args.temperature_moves)
            buffer.extend(samples)
            buffer.extend(mirror(sample) for sample in samples)
        selfplay_time = time.time() - start

        # 2. Train on random mini-batches from the buffer
        model.train()
        losses = [train_step(model, optimizer, buffer, args.batch_size, rng) for _ in range(args.train_steps)]
        policy_loss, value_loss = np.mean(losses, axis=0)

        row = {"iteration": it, "buffer_size": len(buffer), "policy_loss": policy_loss,
               "value_loss": value_loss, "selfplay_time": selfplay_time}

        # 3. Evaluate and save a checkpoint
        if it % args.eval_every == 0 or it == args.iterations:
            row.update(evaluate(model, args))
            torch.save(model.state_dict(), args.out / f"model_{it:03d}.pt")
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
