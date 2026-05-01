"""Turn-processing logic for detected board-state changes.

Human plays White (physical pieces); the robot only executes Black replies via ``try_robot_reply``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import chess

import chess_utils
from move_patterns import BLACK_ID, WHITE_ID
from move_diagnostics import (
    describe_move,
    detect_castling,
    explain_illegal_move,
    print_illegal_move_report,
    print_piece_return_instructions,
    row_col_to_chess_square,
)


@dataclass(frozen=True)
class TurnProcessResult:
    """Outcome of processing one detected board-state change."""

    turn: chess.Color
    prior_board_state: object
    game_over: bool


def process_detected_change(
    chess_board: chess.Board,
    turn: chess.Color,
    board_state,
    change,
    try_robot_reply: Callable[[str], bool],
    on_game_over: Callable[[chess.Board], None],
) -> TurnProcessResult:
    """
    Process one already-detected board change.

    Args:
        chess_board: Current board before applying detected human move.
        turn: Side to move.
        board_state: Latest vision board state.
        change: ``BoardChange`` object from ``move_patterns``.
        try_robot_reply: Runs only when Black is to move: Stockfish + arm. Must not run for White.
    """
    one_removals = change.one_removals
    two_removals = change.two_removals
    one_additions = change.one_additions
    two_additions = change.two_additions

    moving_color_val = change.moving_color_val
    expected_color_val = WHITE_ID if turn == chess.WHITE else BLACK_ID
    if moving_color_val is not None and moving_color_val != expected_color_val:
        wrong_color = "black" if moving_color_val == BLACK_ID else "white"
        right_color = "white" if turn == chess.WHITE else "black"
        print(f"Wrong turn: {wrong_color} moved but it is {right_color}'s turn.")
        if moving_color_val == BLACK_ID and turn == chess.WHITE:
            black_removed = [tuple(rc) for rc in one_removals]
            if black_removed:
                moved_details = []
                for rc in black_removed:
                    sq = row_col_to_chess_square(*rc)
                    piece = chess_board.piece_at(sq)
                    piece_name = chess.piece_name(piece.piece_type) if piece else "unknown piece"
                    moved_details.append(f"{piece_name} from {chess.square_name(sq)}")
                print("Illegal black move on white's turn: " + ", ".join(moved_details))
        print_piece_return_instructions(chess_board, one_removals, two_removals)
        return TurnProcessResult(turn=turn, prior_board_state=board_state, game_over=False)

    if change.is_castle:
        castle_color_val = BLACK_ID if len(one_removals) == 2 else WHITE_ID
        removals = one_removals if castle_color_val == BLACK_ID else two_removals
        additions = one_additions if castle_color_val == BLACK_ID else two_additions
        castle_move = detect_castling(chess_board, removals, additions)
        if castle_move is None:
            print("Looks like castling but no legal castle move found. Please return pieces.")
            return TurnProcessResult(turn=turn, prior_board_state=board_state, game_over=False)

        print(f"Castling detected! UCI move: {castle_move}")
        try:
            chess_board.push_uci(castle_move)
            turn = chess.BLACK if turn == chess.WHITE else chess.WHITE
            print(f"Castling applied: {castle_move}")
        except Exception as exc:
            print(f"  ILLEGAL CASTLE ({exc}): please return pieces to original squares.")
            return TurnProcessResult(turn=turn, prior_board_state=board_state, game_over=False)

        if turn == chess.BLACK and try_robot_reply("Black (robot) move failed"):
            return TurnProcessResult(turn=turn, prior_board_state=board_state, game_over=True)
        return TurnProcessResult(turn=turn, prior_board_state=board_state, game_over=False)

    if not change.is_single_move and not change.is_legal_capture:
        all_removals = [(tuple(rc), BLACK_ID) for rc in one_removals] + [(tuple(rc), WHITE_ID) for rc in two_removals]
        if all_removals:
            print("Invalid board change. Please return pieces to their original squares:")
            print_piece_return_instructions(chess_board, one_removals, two_removals)
        return TurnProcessResult(turn=turn, prior_board_state=board_state, game_over=False)

    move = chess_utils.determine_move(one_removals, two_removals, one_additions, two_additions)
    if turn == chess.WHITE:
        print(f"Recorded human (White) move: {move}")
    else:
        print(f"Detected board change (Black to move): {move}")
    describe_move(chess_board, (one_removals, two_removals), (one_additions, two_additions))

    try:
        legality_reason = explain_illegal_move(chess_board, move)
        if legality_reason != "move is legal":
            raise ValueError(legality_reason)
        chess_board.push_uci(move)
        turn = chess.BLACK if turn == chess.WHITE else chess.WHITE
        if chess_board.is_check():
            checked_side = "White" if chess_board.turn == chess.WHITE else "Black"
            print(f"Check: {checked_side} is in check.")
        if chess_board.is_game_over(claim_draw=True):
            on_game_over(chess_board)
            return TurnProcessResult(turn=turn, prior_board_state=board_state, game_over=True)
    except Exception as exc:
        print(f"  ILLEGAL MOVE: {exc}")
        print(f"  Candidate move was: {move}")
        print_illegal_move_report(chess_board, move)
        print_piece_return_instructions(chess_board, one_removals, two_removals)
        return TurnProcessResult(turn=turn, prior_board_state=board_state, game_over=False)

    if turn == chess.BLACK and try_robot_reply("Black (robot) move failed"):
        return TurnProcessResult(turn=turn, prior_board_state=board_state, game_over=True)

    return TurnProcessResult(turn=turn, prior_board_state=board_state, game_over=False)
