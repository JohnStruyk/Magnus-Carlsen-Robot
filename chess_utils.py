"""Utilities for translating board-state diffs into candidate UCI moves."""

from __future__ import annotations

from typing import Sequence

import chess
import numpy as np


def square_to_alg(square: Sequence[int]) -> str:
    """Convert (row, col) board-state coordinates to algebraic notation."""
    row, col = int(square[0]), int(square[1])
    file_letter = chr(ord("a") + col)
    rank = str(8 - row)
    return f"{file_letter}{rank}"


def _vision_rc_to_sq(rc: Sequence[int]) -> chess.Square:
    r, c = int(rc[0]), int(rc[1])
    return chess.square(c, 7 - r)


def _try_en_passant_uci(
    board: chess.Board,
    mover_rem: np.ndarray,
    mover_add: np.ndarray,
    victim_rem: np.ndarray,
) -> str | None:
    """Map vision diff (mover from/to + captured pawn square) to a legal en-passant UCI, if any."""
    from_sq = _vision_rc_to_sq(mover_rem)
    to_sq = _vision_rc_to_sq(mover_add)
    vic_sq = _vision_rc_to_sq(victim_rem)
    for m in board.legal_moves:
        if not board.is_en_passant(m):
            continue
        if m.from_square != from_sq or m.to_square != to_sq:
            continue
        if board.turn == chess.WHITE:
            cap_sq = m.to_square - 8
        else:
            cap_sq = m.to_square + 8
        if cap_sq == vic_sq:
            return m.uci()
    return None


def _determine_castle(removals: np.ndarray, additions: np.ndarray) -> str | None:
    """Infer kingside/queenside castling from two removals + two additions."""
    if removals.shape[0] != 2 or additions.shape[0] != 2:
        return None

    removal_cols = sorted(int(x) for x in removals[:, 1])
    addition_cols = sorted(int(x) for x in additions[:, 1])
    if removal_cols != [0, 4] and removal_cols != [4, 7]:
        return None
    if addition_cols == [5, 6]:
        return "O-O"
    if addition_cols == [2, 3]:
        return "O-O-O"
    return None


def _determine_capture(
    capturing_removal: Sequence[int],
    capturing_addition: Sequence[int],
    captured_removal: Sequence[int],
) -> str | None:
    """Infer capture move if destination equals captured piece square."""
    if np.array_equal(capturing_addition, captured_removal):
        return square_to_alg(capturing_removal) + square_to_alg(capturing_addition)
    return None


def _determine_normal_move(moving_removal: Sequence[int], moving_addition: Sequence[int]) -> str:
    """Infer non-capture move from one removal + one addition."""
    return square_to_alg(moving_removal) + square_to_alg(moving_addition)


def determine_move(
    one_removals,
    two_removals,
    one_additions,
    two_additions,
    board: chess.Board | None = None,
) -> str:
    """
    Convert board-state color deltas into a candidate move string.

    Pass ``board`` so en passant (capturing pawn lands on empty square) can be resolved.

    Returns:
        - UCI string for normal/capture/en-passant moves (e.g. ``e5d6``)
        - SAN castle shorthand for castles (``O-O`` or ``O-O-O``)
        - ``BAD. ...`` diagnostic string when the diff is inconsistent
    """
    if len(one_additions) + len(two_additions) > len(one_removals) + len(two_removals):
        return "BAD. there are more pieces now than at start of move"

    if len(one_additions) + len(two_additions) + 1 < len(one_removals) + len(two_removals):
        return "BAD. too many pieces removed"

    if len(one_additions) > 0 and len(two_additions) > 0:
        return "BAD. both color pieces have moved."

    if len(one_removals) + len(two_removals) == 0:
        return "BAD. neither color piece has moved."

    if len(one_removals) == 1 and len(one_additions) == 1:
        if len(two_removals) == 1:
            capture = _determine_capture(one_removals[0], one_additions[0], two_removals[0])
            if capture is not None:
                return capture
            if board is not None:
                ep = _try_en_passant_uci(board, one_removals[0], one_additions[0], two_removals[0])
                if ep is not None:
                    return ep
            return "BAD. capturing piece did not end on captured square (and not en passant)"
        return _determine_normal_move(one_removals[0], one_additions[0])

    if len(one_removals) == 2 and len(one_additions) == 2:
        castle = _determine_castle(one_removals, one_additions)
        return castle if castle is not None else "BAD. castling pattern not recognized"

    if len(two_removals) == 1 and len(two_additions) == 1:
        if len(one_removals) == 1:
            capture = _determine_capture(two_removals[0], two_additions[0], one_removals[0])
            if capture is not None:
                return capture
            if board is not None:
                ep = _try_en_passant_uci(board, two_removals[0], two_additions[0], one_removals[0])
                if ep is not None:
                    return ep
            return "BAD. capturing piece did not end on captured square (and not en passant)"
        return _determine_normal_move(two_removals[0], two_additions[0])

    if len(two_removals) == 2 and len(two_additions) == 2:
        castle = _determine_castle(two_removals, two_additions)
        return castle if castle is not None else "BAD. castling pattern not recognized"

    return "BAD. unsupported change pattern"
