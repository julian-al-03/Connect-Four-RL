import numpy as np
import pytest
from pettingzoo.classic import connect_four_v3

from connect4.game import ConnectFour


def play_moves(moves):
    state = ConnectFour()
    for col in moves:
        state = state.play(col)
    return state


@pytest.mark.parametrize("moves", [
    [0, 0, 1, 1, 2, 2, 3],              # horizontal
    [0, 1, 0, 1, 0, 1, 0],              # vertical
    [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3],  # diagonal /
    [6, 5, 5, 4, 4, 3, 4, 3, 3, 0, 3],  # diagonal \
])
def test_wins(moves):
    state = play_moves(moves[:-1])
    assert not state.is_terminal()
    state = state.play(moves[-1])
    assert state.is_terminal()
    assert state.terminal_value() == -1.0  # player to move has lost


def test_play_does_not_modify_state():
    state = ConnectFour()
    state.play(3)
    assert state.num_moves == 0 and not state.board.any()


def test_full_column_is_illegal():
    state = play_moves([0] * 6)
    assert not state.legal_moves()[0]
    with pytest.raises(ValueError):
        state.play(0)


def test_matches_pettingzoo_on_random_games():
    rng = np.random.default_rng(0)
    env = connect_four_v3.env()
    for seed in range(300):
        env.reset(seed=seed)
        state = ConnectFour()
        final_rewards = {}
        for agent in env.agent_iter():
            obs, reward, termination, truncation, _ = env.last()
            if termination or truncation:
                assert state.is_terminal()
                final_rewards[agent] = reward
                env.step(None)
                continue
            assert not state.is_terminal()
            # PettingZoo is (rows, cols, planes); ours is (planes, rows, cols)
            np.testing.assert_array_equal(state.encode(), obs["observation"].transpose(2, 0, 1))
            np.testing.assert_array_equal(state.legal_moves(), obs["action_mask"].astype(bool))
            col = int(rng.choice(np.flatnonzero(obs["action_mask"])))
            env.step(col)
            state = state.play(col)
        # terminal_value() is from the view of the player to move; state.player converts
        # it to player_0's view (+1 if player_0 is to move, -1 otherwise)
        assert final_rewards["player_0"] == state.terminal_value() * state.player
