"""Persistence and startup-board helpers for the game loop."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import chess
import numpy as np

from stockfish_int import visualize_board


SAVED_GAME_FILE = "stored_game.txt"
STANDARD_BOARD_STATE = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2],
    ],
    dtype=int,
)


def detect_starting_fen(board_state: np.ndarray, current_fen: str) -> Optional[str]:
    """Infer initial FEN from vision state; fallback to configured FEN if provided."""
    if np.array_equal(board_state, STANDARD_BOARD_STATE):
        print("Standard starting position detected.")
        return chess.STARTING_FEN
    if current_fen:
        print("Using CURRENT_FEN.")
        return current_fen
    return None


def save_game_state(chess_board: Optional[chess.Board]) -> None:
    """Persist board FEN to disk when game exits."""
    if chess_board is None:
        return
    with open(SAVED_GAME_FILE, "w", encoding="utf-8") as output_file:
        output_file.write(chess_board.fen())
    print(f"\nGame saved to {SAVED_GAME_FILE}")


def load_saved_game() -> Optional[str]:
    """Load saved FEN from disk if available."""
    if not os.path.exists(SAVED_GAME_FILE):
        return None
    with open(SAVED_GAME_FILE, "r", encoding="utf-8") as input_file:
        return input_file.read().strip()


def show_board_and_wait(chess_board: chess.Board) -> None:
    """Open current board SVG and wait for user confirmation."""
    visualize_board(chess_board)
    input("Board displayed in browser. Press Enter to start the game loop...")
    if os.path.exists("current_board.svg"):
        os.remove("current_board.svg")


def prompt_continue_saved_game() -> Tuple[Optional[chess.Board], bool]:
    """Prompt user to resume saved game if a FEN file is available."""
    saved_fen = load_saved_game()
    if saved_fen is None:
        return None, False

    print(f"\nSaved game found in '{SAVED_GAME_FILE}'.")
    print(f"  FEN: {saved_fen}")
    answer = input("Continue saved game? [y/N]: ").strip().lower()
    if answer == "y":
        board = chess.Board(saved_fen)
        print("Resuming saved game.")
        show_board_and_wait(board)
        return board, True

    os.remove(SAVED_GAME_FILE)
    print("Saved game deleted. Starting fresh.")
    return None, False
