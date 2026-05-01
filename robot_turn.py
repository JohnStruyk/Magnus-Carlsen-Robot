"""Robot move execution helpers used by the game loop."""

from __future__ import annotations

import time
from typing import Callable, Tuple

import chess

from piece_continuity import get_board_state
from pickup_board_piece import (
    capture_piece,
    move_piece,
    place_promotion_queen_from_source,
    remove_piece_to_graveyard,
)
from stockfish_int import get_best_move


def count_white_captures(chess_board: chess.Board) -> int:
    """Return how many white pieces have been captured from the initial 16."""
    white_pieces_on_board = sum(1 for piece in chess_board.piece_map().values() if piece.color == chess.WHITE)
    return 16 - white_pieces_on_board


def execute_robot_move_on_board(chess_board: chess.Board, robot_move: chess.Move, zed) -> str:
    """Execute one physical move (normal, capture, castling, promotion replacement)."""
    move_string = robot_move.uci()
    print(f"Sending move {move_string} to robot arm...")

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

    from_sq = robot_move.from_square
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
        remove_piece_to_graveyard(from_occupant, to_square, zed, capture_count=0)
        place_promotion_queen_from_source(to_square, zed)

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
            time.sleep(0.5)


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
) -> Tuple[int, object, bool]:
    """Execute engine move for black and return ``(turn, prior_board_state, game_over)``."""
    robot_move = get_best_move(chess_board.fen(), time_limit=2.0)
    move_string = execute_robot_move_with_retry(chess_board, robot_move, zed)
    chess_board.push_uci(move_string)

    turn = chess.WHITE
    if chess_board.is_check():
        checked_side = "White" if chess_board.turn == chess.WHITE else "Black"
        print(f"Check: {checked_side} is in check.")
    if chess_board.is_game_over(claim_draw=True):
        on_game_over(chess_board)
        return turn, capture_stable_board_state(zed, detector, camera_intrinsic), True

    prior_board_state = capture_stable_board_state(zed, detector, camera_intrinsic)
    print(f"Robot played {move_string}. White's turn.")
    return turn, prior_board_state, False
