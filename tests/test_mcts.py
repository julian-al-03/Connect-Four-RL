import numpy as np

from connect4.game import ConnectFour
from connect4.mcts import MCTS, MCTSAgent, RolloutEvaluator, visit_counts


def play_moves(moves):
    state = ConnectFour()
    for col in moves:
        state = state.play(col)
    return state


def make_agent(seed=0, sims=200):
    return MCTSAgent(MCTS(RolloutEvaluator(seed), num_simulations=sims))


def test_visits_only_legal_moves_and_add_up():
    state = play_moves([0] * 6)  # column 0 is full
    root = MCTS(RolloutEvaluator(0), num_simulations=100).search(state)
    counts = visit_counts(root)
    assert counts[0] == 0
    assert counts.sum() == 99  # the first simulation only expands the root


def test_takes_immediate_win():
    state = play_moves([0, 1, 0, 1, 0, 1])
    assert make_agent()(state) == 0


def test_blocks_immediate_loss():
    state = play_moves([0, 1, 0, 1, 0])
    assert make_agent()(state) == 0


def test_winning_move_has_high_q():
    state = play_moves([0, 1, 0, 1, 0, 1])
    root = MCTS(RolloutEvaluator(0), num_simulations=200).search(state)
    # moving into a won position is worth +1 for the player who moved
    assert np.isclose(root.children[0].q(), 1.0)


def test_terminal_nodes_find_tactics_without_any_evaluation_signal():
    # An evaluator that knows nothing (uniform priors, value 0): immediate wins and losses
    # are still found, because terminal nodes in the tree return the true result.
    def clueless(state):
        legal = state.legal_moves()
        return legal / legal.sum(), 0.0

    agent = MCTSAgent(MCTS(clueless, num_simulations=100))
    assert agent(play_moves([0, 1, 0, 1, 0, 1])) == 0  # win
    assert agent(play_moves([0, 1, 0, 1, 0])) == 0     # block


def test_dirichlet_noise_changes_root_priors_only_when_requested():
    mcts = MCTS(RolloutEvaluator(0), num_simulations=50, seed=0)
    state = play_moves([0] * 6)  # column 0 is full
    plain = [child.prior for child in mcts.search(state).children.values()]
    noisy_root = mcts.search(state, add_noise=True)
    noisy = [child.prior for child in noisy_root.children.values()]
    assert np.allclose(plain, 1 / 6)
    assert not np.allclose(noisy, plain)
    assert np.isclose(sum(noisy), 1.0)
    assert 0 not in noisy_root.children  # noise only goes to legal moves
