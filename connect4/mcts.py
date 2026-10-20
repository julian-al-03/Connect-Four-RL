"""Monte Carlo Tree Search with PUCT selection.

The search is independent of how leaves are evaluated: an evaluator maps a state to
(priors, value), where priors is a distribution over the 7 columns and value is the
expected result for the player to move. Pure MCTS uses RolloutEvaluator (uniform priors,
random playout); AlphaZero will use the policy-value network instead.
"""
import math

import numpy as np

from connect4.game import COLS


class Node:
    """A node in the search tree.

    visits and value_sum are from the view of the player who made the move into this node,
    so a parent can compare its children directly: higher Q is better for the parent's player.
    """
    __slots__ = ("state", "prior", "children", "visits", "value_sum")

    def __init__(self, state, prior):
        self.state = state  # created lazily when the node is first selected
        self.prior = prior
        self.children = {}  # column -> Node, empty until expanded
        self.visits = 0
        self.value_sum = 0.0

    def q(self):
        return self.value_sum / self.visits if self.visits else 0.0


class RolloutEvaluator:
    """Classic MCTS evaluation: uniform priors and the result of one random playout."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, state):
        legal = state.legal_moves()
        priors = legal / legal.sum()
        start_player = state.player
        while not state.is_terminal():
            state = state.play(self.rng.choice(np.flatnonzero(state.legal_moves())))
        # terminal_value() is from the view of the player to move at the end;
        # multiplying by both players' signs converts it to start_player's view
        return priors, state.terminal_value() * state.player * start_player


class MCTS:
    def __init__(self, evaluate, num_simulations=200, c_puct=1.5,
                 dirichlet_alpha=1.0, dirichlet_eps=0.25, seed=None):
        self.evaluate = evaluate
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = np.random.default_rng(seed)

    def search(self, state, add_noise=False):
        """Run the simulations from `state` and return the root node.

        add_noise=True mixes Dirichlet noise into the root priors (self-play only).
        """
        root = Node(state, prior=1.0)
        self._simulate(root)  # the first simulation expands the root
        if add_noise:
            self._add_dirichlet_noise(root)
        for _ in range(self.num_simulations - 1):
            self._simulate(root)
        return root

    def _simulate(self, root):
        node, path = root, [root]
        # 1. Selection: descend until we reach a node that is not expanded yet
        while node.children:
            node = self._select_child(node)
            path.append(node)
        # 2. Expansion + 3. Evaluation
        if node.state.is_terminal():
            value = node.state.terminal_value()
        else:
            priors, value = self.evaluate(node.state)
            for col in np.flatnonzero(node.state.legal_moves()):
                node.children[int(col)] = Node(None, prior=priors[col])
        # 4. Backpropagation. `value` is from the view of the player to move at the leaf;
        # each node stores values from the view of the player who moved into it, so the
        # sign flips at every level (negamax).
        for n in reversed(path):
            value = -value
            n.value_sum += value
            n.visits += 1

    def _add_dirichlet_noise(self, root):
        """P(s0, a) = (1 - eps) * p_a + eps * eta_a with eta ~ Dir(alpha), over the legal moves.

        Makes self-play try moves the network currently dislikes, so it can discover
        that some of them are good.
        """
        children = list(root.children.values())
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(children))
        for child, eta in zip(children, noise):
            child.prior = (1 - self.dirichlet_eps) * child.prior + self.dirichlet_eps * eta

    def _select_child(self, node):
        """PUCT: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))."""
        # The paper uses sqrt(sum of child visits); node.visits is that plus one (the visit
        # that expanded the node), which avoids a zero exploration term on the first selection.
        sqrt_n = math.sqrt(node.visits)
        col, child = max(
            node.children.items(),
            key=lambda item: item[1].q() + self.c_puct * item[1].prior * sqrt_n / (1 + item[1].visits),
        )
        if child.state is None:
            child.state = node.state.play(col)
        return child


def visit_counts(root):
    counts = np.zeros(COLS)
    for col, child in root.children.items():
        counts[col] = child.visits
    return counts


class MCTSAgent:
    """Plays the most visited move after searching."""

    def __init__(self, mcts):
        self.mcts = mcts

    def __call__(self, state):
        return int(np.argmax(visit_counts(self.mcts.search(state))))
