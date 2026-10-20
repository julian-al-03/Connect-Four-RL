"""Play agents against each other and collect statistics.

Run `python -m connect4.arena` for a table of the baseline matchups.
"""
from connect4.agents import GreedyAgent, RandomAgent, pure_mcts_agent
from connect4.game import ConnectFour


def play_game(first, second):
    """Play one game. Returns +1 if `first` wins, -1 if `second` wins, 0 for a draw."""
    agents = {1: first, -1: second}
    state = ConnectFour()
    while not state.is_terminal():
        state = state.play(agents[state.player](state))
    # terminal_value() is from the view of the player to move; state.player converts it
    # to the first player's view
    return int(state.terminal_value() * state.player)


def play_match(agent_a, agent_b, n_games):
    """Play n_games, alternating who moves first. Returns results from agent_a's view."""
    stats = {"wins": 0, "draws": 0, "losses": 0,
             "wins_as_first": 0, "wins_as_second": 0}
    for g in range(n_games):
        a_first = g % 2 == 0
        result = play_game(agent_a, agent_b) if a_first else -play_game(agent_b, agent_a)
        if result == 1:
            stats["wins"] += 1
            stats["wins_as_first" if a_first else "wins_as_second"] += 1
        elif result == 0:
            stats["draws"] += 1
        else:
            stats["losses"] += 1
    return stats


if __name__ == "__main__":
    matchups = [  # (name A, agent A, name B, agent B, games)
        ("random", RandomAgent(0), "random", RandomAgent(1), 1000),
        ("greedy", GreedyAgent(0), "random", RandomAgent(1), 1000),
        ("greedy", GreedyAgent(0), "greedy", GreedyAgent(1), 1000),
        ("mcts200", pure_mcts_agent(200, 0), "random", RandomAgent(1), 100),
        ("mcts200", pure_mcts_agent(200, 0), "greedy", GreedyAgent(1), 100),
        ("mcts1000", pure_mcts_agent(1000, 0), "mcts200", pure_mcts_agent(200, 1), 100),
    ]
    print("Who starts alternates; 'as 1st'/'as 2nd' split A's wins by colour.\n")
    print(f"{'A':>8} vs {'B':<8} {'games':>6} {'A wins':>7} {'draws':>6} {'B wins':>7} "
          f"{'A wins as 1st':>14} {'A wins as 2nd':>14}")
    for name_a, a, name_b, b, n in matchups:
        s = play_match(a, b, n)
        print(f"{name_a:>8} vs {name_b:<8} {n:>6} {s['wins']:>7} {s['draws']:>6} {s['losses']:>7} "
              f"{s['wins_as_first']:>14} {s['wins_as_second']:>14}", flush=True)
