from connect4.agents import GreedyAgent, RandomAgent
from connect4.arena import play_game, play_match
from connect4.game import ConnectFour


def play_moves(moves):
    state = ConnectFour()
    for col in moves:
        state = state.play(col)
    return state


def test_random_agent_plays_legal_moves():
    state = play_moves([0] * 6)  # column 0 is full
    agent = RandomAgent(seed=0)
    assert all(agent(state) != 0 for _ in range(100))


def test_greedy_takes_immediate_win():
    state = play_moves([0, 1, 0, 1, 0, 1])  # first player has three in column 0
    assert GreedyAgent(seed=0)(state) == 0


def test_greedy_blocks_immediate_loss():
    state = play_moves([0, 1, 0, 1, 0])  # second player to move, first player threatens column 0
    assert GreedyAgent(seed=0)(state) == 0


def test_play_game_result_perspective():
    always_col = lambda col: (lambda state: col)
    # first player stacks column 0, second stacks column 1: first player wins
    assert play_game(always_col(0), always_col(1)) == 1


def test_play_match_counts_add_up():
    s = play_match(RandomAgent(0), RandomAgent(1), 20)
    assert s["wins"] + s["draws"] + s["losses"] == 20
    assert s["wins_as_first"] + s["wins_as_second"] == s["wins"]
