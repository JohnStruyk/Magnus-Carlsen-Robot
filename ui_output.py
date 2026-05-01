"""Terminal output helpers for game status messages."""

from __future__ import annotations

import sys
import time

import chess


def print_game_over_banner(chess_board: chess.Board) -> None:
    """Print an animated game-over report for terminal visibility."""
    outcome = chess_board.outcome(claim_draw=True)
    termination = outcome.termination.name.replace("_", " ").title() if outcome else "Unknown"
    result = outcome.result() if outcome else chess_board.result(claim_draw=True)

    if result == "1-0":
        winner_line = "Winner: White"
    elif result == "0-1":
        winner_line = "Winner: Black"
    else:
        winner_line = "Winner: None (Draw)"

    if chess_board.is_checkmate():
        reason_line = "Reason: Checkmate delivered. The side to move has no legal escape."
    elif chess_board.is_stalemate():
        reason_line = "Reason: Stalemate. No legal moves remain, but the king is not in check."
    elif chess_board.is_insufficient_material():
        reason_line = "Reason: Draw by insufficient material."
    elif chess_board.is_fifty_moves():
        reason_line = "Reason: Draw by fifty-move rule."
    elif chess_board.is_repetition(3):
        reason_line = "Reason: Draw by threefold repetition."
    else:
        reason_line = f"Reason: {termination}."

    def _type_line(text: str, delay_s: float = 0.01) -> None:
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(delay_s)
        sys.stdout.write("\n")
        sys.stdout.flush()

    frames = [
        ("#" * 72, "GAME OVER".center(72), "#" * 72),
        ("*" * 72, "CHECKMATE CHRONICLE".center(72), "*" * 72),
        ("=" * 72, "FINAL POSITION LOCKED".center(72), "=" * 72),
    ]

    print()
    for top, middle, bottom in frames:
        print(top)
        print(middle)
        print(bottom)
        time.sleep(0.12)

    _type_line(f"Final result      : {result}", delay_s=0.008)
    _type_line(f"{winner_line}", delay_s=0.008)
    _type_line(f"{reason_line}", delay_s=0.006)
    _type_line(f"Termination type  : {termination}", delay_s=0.008)
    _type_line(f"Final FEN         : {chess_board.fen()}", delay_s=0.003)
    print("=" * 72 + "\n")
