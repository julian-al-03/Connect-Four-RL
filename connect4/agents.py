"""Baseline agents. An agent is any callable that maps a ConnectFour state to a column."""
import numpy as np

from connect4.mcts import MCTS, MCTSAgent, RolloutEvaluator


class RandomAgent:
    """Picks a uniformly random legal column."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, state):
        return int(self.rng.choice(np.flatnonzero(state.legal_moves())))


class GreedyAgent:
    """One-step lookahead: win if possible, else block the opponent's immediate win, else random."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, state):
        legal = np.flatnonzero(state.legal_moves())
        for col in legal:
            if state.play(col).last_move_won:
                return int(col)
        # Pretend it is the opponent's turn to find the columns where they would win next move
        opponent_view = state.copy()
        opponent_view.player = -state.player
        for col in legal:
            if opponent_view.play(col).last_move_won:
                return int(col)
        return int(self.rng.choice(legal))


def pure_mcts_agent(num_simulations, seed=None):
    """MCTS baseline without a network: uniform priors and random rollouts."""
    return MCTSAgent(MCTS(RolloutEvaluator(seed), num_simulations=num_simulations))
