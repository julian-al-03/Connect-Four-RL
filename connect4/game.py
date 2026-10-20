"""Lightweight Connect Four game state for tree search.

PettingZoo's environment is convenient for playing full games, but MCTS has to copy
the game state thousands of times per move, and deep-copying the environment is slow.
This class keeps only what the search needs: a NumPy board, the column heights and
the player to move.
"""
import numpy as np

ROWS, COLS = 6, 7
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))  # horizontal, vertical, two diagonals


class ConnectFour:
    """A Connect Four position. play() returns a new state and leaves this one unchanged.

    board:  (ROWS, COLS) int8, row 0 is the top (as in PettingZoo).
            +1 = first player's pieces, -1 = second player's, 0 = empty.
    player: +1 if the first player is to move, -1 otherwise.
    """

    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.heights = np.zeros(COLS, dtype=np.int8)  # number of pieces per column
        self.player = 1
        self.num_moves = 0
        self.last_move_won = False

    def copy(self):
        new = ConnectFour.__new__(ConnectFour)
        new.board = self.board.copy()
        new.heights = self.heights.copy()
        new.player = self.player
        new.num_moves = self.num_moves
        new.last_move_won = self.last_move_won
        return new

    def legal_moves(self):
        """Boolean mask of length COLS, True where the column is not full."""
        return self.heights < ROWS

    def play(self, col):
        """Return the state after the player to move drops a piece in column `col`."""
        if self.is_terminal() or self.heights[col] >= ROWS:
            raise ValueError(f"illegal move {col}")
        new = self.copy()
        row = ROWS - 1 - new.heights[col]
        new.board[row, col] = new.player
        new.heights[col] += 1
        new.num_moves += 1
        new.last_move_won = new._is_win(row, col)
        new.player = -new.player
        return new

    def _is_win(self, row, col):
        """Check whether the piece at (row, col) completes four in a row.

        Only lines through the last move can be new wins, so we never scan the whole board.
        """
        piece = self.board[row, col]
        for dr, dc in DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == piece:
                    count += 1
                    r, c = r + sign * dr, c + sign * dc
            if count >= 4:
                return True
        return False

    def is_terminal(self):
        return self.last_move_won or self.num_moves == ROWS * COLS

    def terminal_value(self):
        """Result of a finished game from the perspective of the player to move.

        Only the player who just moved can have won, so this is -1 (loss) or 0 (draw).
        """
        return -1.0 if self.last_move_won else 0.0

    def encode(self):
        """Network input of shape (2, ROWS, COLS), float32.

        Plane 0 = pieces of the player to move, plane 1 = opponent's pieces. Same content
        as PettingZoo's observation, but channels-first as PyTorch expects.
        """
        return np.stack([self.board == self.player, self.board == -self.player]).astype(np.float32)

    def __str__(self):
        symbols = {1: "X", -1: "O", 0: "."}
        rows = [" ".join(symbols[v] for v in row) for row in self.board]
        return "\n".join(rows + [" ".join(str(c) for c in range(COLS))])
