"""Play Connect Four against an agent in the terminal.

Example:
    python -m connect4.play cnn:runs/main/cnn_seed0/model_100.pt --simulations 100

Agents are given as in connect4.evaluate, e.g. greedy, pure:400 or mlp:CHECKPOINT:0 (no search).
"""
import argparse

from connect4.evaluate import make_agent
from connect4.game import COLS, ConnectFour


def read_move(state):
    """Ask for a column until the input is a legal move."""
    legal = state.legal_moves()
    while True:
        text = input(f"Your move (0-{COLS - 1}): ").strip()
        if text.isdigit() and int(text) < COLS and legal[int(text)]:
            return int(text)
        print("That is not a legal move.")


def play_against(agent, human_first=True):
    """Play one game against `agent`. Returns +1 if you win, -1 if you lose, 0 for a draw.

    The first player's pieces are X, the second player's are O.
    """
    human = 1 if human_first else -1
    print(f"You play {'X' if human_first else 'O'}.")
    state = ConnectFour()
    while not state.is_terminal():
        if state.player == human:
            print(f"\n{state}\n")
            move = read_move(state)
        else:
            move = agent(state)
            print(f"Agent plays column {move}.")
        state = state.play(move)
    print(f"\n{state}\n")
    # terminal_value() is from the view of the player to move; state.player converts it
    # to the first player's view and `human` to yours
    result = int(state.terminal_value() * state.player * human)
    print({1: "You win!", 0: "Draw.", -1: "You lose."}[result])
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("agent")
    p.add_argument("--simulations", type=int, default=100, help="simulations per move for a network agent")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--second", action="store_true", help="let the agent make the first move")
    args = p.parse_args()
    play_against(make_agent(args.agent, args.simulations, args.c_puct, seed=None), human_first=not args.second)
