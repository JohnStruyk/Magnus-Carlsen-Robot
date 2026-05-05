"""Spawn Stockfish over UCI. Can dump SVG when the operator sanity-checks FEN."""

import os
import webbrowser
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import chess.svg


DEFAULT_STOCKFISH_PATH = os.environ.get(
    "STOCKFISH_PATH",
    "/home/rob/Desktop/stockfish/stockfish-ubuntu-x86-64-avx2",
)
BOARD_SVG_PATH = "current_board.svg"


def get_best_move(fen_string: str, time_limit: float = 2.0, engine_path: Optional[str] = None) -> chess.Move:
    engine_executable = engine_path or DEFAULT_STOCKFISH_PATH
    if not Path(engine_executable).exists():
        raise FileNotFoundError(
            f"Stockfish executable not found at '{engine_executable}'. "
            "Set STOCKFISH_PATH or pass engine_path explicitly."
        )

    board = chess.Board(fen_string)
    with chess.engine.SimpleEngine.popen_uci(engine_executable) as engine:
        result = engine.play(board, chess.engine.Limit(time=float(time_limit)))
    return result.move


def visualize_board(board: chess.Board, best_move: Optional[chess.Move] = None, output_path: str = BOARD_SVG_PATH) -> None:
    arrows = []
    if best_move is not None:
        arrows = [chess.svg.Arrow(best_move.from_square, best_move.to_square, color="#0000cccc")]

    board_svg = chess.svg.board(board=board, arrows=arrows, size=400)
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(board_svg)
    webbrowser.open("file://" + os.path.realpath(output_path))
