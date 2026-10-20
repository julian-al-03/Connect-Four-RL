import numpy as np

from connect4.mcts import MCTS, RolloutEvaluator
from connect4.selfplay import mirror, self_play_game


def play_game(seed=0):
    mcts = MCTS(RolloutEvaluator(seed), num_simulations=20, seed=seed)
    return self_play_game(mcts, np.random.default_rng(seed))


def test_samples_are_valid():
    for s, pi, z in play_game():
        assert s.shape == (2, 6, 7)
        assert np.isclose(pi.sum(), 1.0)
        full_columns = s[:, 0, :].sum(axis=0) > 0
        assert np.all(pi[full_columns] == 0)
        assert z in (-1, 0, 1)


def test_value_targets_alternate_and_last_mover_wins():
    for seed in range(5):
        samples = play_game(seed)
        z = [sample[2] for sample in samples]
        # consecutive positions have different players to move, so their z have opposite signs
        assert all(a == -b for a, b in zip(z, z[1:]))
        # the player who made the last move won (unless the game is a draw)
        assert z[-1] in (1, 0)
        assert len(samples) == np.count_nonzero(samples[-1][0]) + 1  # one sample per move


def test_mirror():
    s, pi, z = play_game()[5]
    ms, mpi, mz = mirror((s, pi, z))
    assert np.array_equal(ms[:, :, 0], s[:, :, 6])
    assert np.array_equal(mpi, pi[::-1]) and mz == z
