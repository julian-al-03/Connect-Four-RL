"""Self-play: the agent plays against itself and records training samples (s, pi, z)."""
import numpy as np

from connect4.game import ConnectFour
from connect4.mcts import visit_counts


def self_play_game(mcts, rng, temperature_moves=10):
    """Play one game with MCTS on both sides. Returns a list of (encoded state, pi, z).

    pi: normalized root visit counts (the policy target).
    z:  final result from the view of the player to move in that state (the value target).
    For the first `temperature_moves` moves, the move is sampled from pi (tau = 1) so that
    games differ; afterwards the most visited move is played (tau -> 0).
    """
    state = ConnectFour()
    history = []  # (encoded state, pi, player to move)
    while not state.is_terminal():
        root = mcts.search(state, add_noise=True)
        pi = visit_counts(root)
        pi /= pi.sum()
        history.append((state.encode(), pi, state.player))
        if state.num_moves < temperature_moves:
            move = rng.choice(len(pi), p=pi)
        else:
            move = int(np.argmax(pi))
        state = state.play(move)
    # Result from the first player's view (+1, 0 or -1), then per sample from the mover's view
    first_player_result = state.terminal_value() * state.player
    return [(s, pi, first_player_result * player) for s, pi, player in history]


def mirror(sample):
    """Connect Four is left-right symmetric: mirroring the board and pi gives another valid sample."""
    s, pi, z = sample
    return s[:, :, ::-1].copy(), pi[::-1].copy(), z
