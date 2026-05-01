"""Robot move execution helpers used by the game loop (Black only — human plays White)."""

from __future__ import annotations

import time
from typing import Callable, Tuple

import chess

from piece_continuity import get_board_state
from pickup_board_piece import (
    capture_piece,
    move_piece,
    replace_promoted_pawn_with_source_queen,
)
from stockfish_int import get_best_move


def _assert_black_to_move(chess_board: chess.Board, context: str) -> None:
    if chess_board.turn != chess.BLACK:
        raise RuntimeError(f"{context}: robot only moves Black; White to move — refusing arm motion.")


def count_white_captures(chess_board: chess.Board) -> int:
    """Return how many white pieces have been captured from the initial 16."""
    white_pieces_on_board = sum(1 for piece in chess_board.piece_map().values() if piece.color == chess.WHITE)
    return 16 - white_pieces_on_board


def execute_robot_move_on_board(chess_board: chess.Board, robot_move: chess.Move, zed) -> str:
    """Execute one physical move for the side to move (must be Black)."""
    _assert_black_to_move(chess_board, "execute_robot_move_on_board")
    from_sq = robot_move.from_square
    piece = chess_board.piece_at(from_sq)
    if piece is None or piece.color != chess.BLACK:
        raise RuntimeError(
            f"Refusing arm motion: expected Black piece on {chess.square_name(from_sq)}, "
            "robot never moves White."
        )
    move_string = robot_move.uci()
    print(f"Sending Black move {move_string} to robot arm...")

    if chess_board.is_castling(robot_move):
        king_from_sq = robot_move.from_square
        king_to_sq = robot_move.to_square
        king_piece = chess_board.piece_at(king_from_sq)
        if king_piece is None:
            raise RuntimeError("Castling failed: king piece not found on from-square.")

        if chess_board.is_kingside_castling(robot_move):
            rook_from_sq = chess.H1 if chess_board.turn == chess.WHITE else chess.H8
            rook_to_sq = chess.F1 if chess_board.turn == chess.WHITE else chess.F8
        else:
            rook_from_sq = chess.A1 if chess_board.turn == chess.WHITE else chess.A8
            rook_to_sq = chess.D1 if chess_board.turn == chess.WHITE else chess.D8

        rook_piece = chess_board.piece_at(rook_from_sq)
        if rook_piece is None:
            raise RuntimeError("Castling failed: rook piece not found on from-square.")

        move_piece(king_piece.symbol(), chess.square_name(king_from_sq), chess.square_name(king_to_sq), zed)
        move_piece(rook_piece.symbol(), chess.square_name(rook_from_sq), chess.square_name(rook_to_sq), zed)
        return move_string

    to_sq = robot_move.to_square
    from_piece = chess_board.piece_at(from_sq)
    to_piece = chess_board.piece_at(to_sq)
    if from_piece is None:
        raise RuntimeError("Robot move failed: source square has no piece.")

    from_square = chess.square_name(from_sq)
    to_square = chess.square_name(to_sq)
    from_occupant = from_piece.symbol()
    to_occupant = to_piece.symbol() if to_piece is not None else None

    if to_occupant is not None:
        capture_count = count_white_captures(chess_board)
        capture_piece(from_occupant, to_occupant, from_square, to_square, zed, capture_count)
    else:
        move_piece(from_occupant, from_square, to_square, zed)

    if robot_move.promotion is not None:
        replace_promoted_pawn_with_source_queen(to_square, zed)

    return move_string


def execute_robot_move_with_retry(chess_board: chess.Board, robot_move: chess.Move, zed) -> str:
    """Retry robot move indefinitely for any non-interrupt exception."""
    attempt = 1
    while True:
        try:
            return execute_robot_move_on_board(chess_board, robot_move, zed)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            from_name = chess.square_name(robot_move.from_square)
            to_name = chess.square_name(robot_move.to_square)
            print(
                f"Robot move retry #{attempt} for {robot_move.uci()} "
                f"(source {from_name} -> destination {to_name}): {exc}"
            )
            attempt += 1
            time.sleep(3)


def capture_stable_board_state(zed, detector, camera_intrinsic):
    """Read camera until a non-None board state is available."""
    stable_state = None
    while stable_state is None:
        post_robot_image = zed.image
        stable_state, _, _ = get_board_state(post_robot_image, detector, camera_intrinsic)
        if stable_state is None:
            print("Post-robot board capture failed; retrying in 0.5s...")
            time.sleep(0.5)
    return stable_state


def execute_robot_reply_turn(
    chess_board: chess.Board,
    zed,
    detector,
    camera_intrinsic,
    on_game_over: Callable[[chess.Board], None],
) -> Tuple[chess.Color, object, bool]:
    """Ask Stockfish for Black's move, execute it on the arm, return ``(side_to_move_after, prior_board_state, game_over)``."""
    _assert_black_to_move(chess_board, "execute_robot_reply_turn")
    robot_move = get_best_move(chess_board.fen(), time_limit=2.0)
    if chess_board.color_at(robot_move.from_square) != chess.BLACK:
        raise RuntimeError("Engine suggested a non-Black move; robot never plays White.")
    move_string = execute_robot_move_with_retry(chess_board, robot_move, zed)
    chess_board.push_uci(move_string)

    if chess_board.is_check():
        checked_side = "White" if chess_board.turn == chess.WHITE else "Black"
        print(f"Check: {checked_side} is in check.")
    if chess_board.is_game_over(claim_draw=True):
        on_game_over(chess_board)
        return chess_board.turn, capture_stable_board_state(zed, detector, camera_intrinsic), True

    prior_board_state = capture_stable_board_state(zed, detector, camera_intrinsic)
    print(f"Robot (Black) played {move_string}. Human (White) to move.")
    return chess_board.turn, prior_board_state, False
