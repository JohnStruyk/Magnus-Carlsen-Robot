"""
Manipulation stress test — moves a piece back and forth 5 times (10 total manipulations).
Each call to move_piece handles vision, arm connect/disconnect, and graveyard return internally.

Edit the three constants below to configure the test.
"""

from __future__ import annotations

import argparse

from utils.zed_camera import ZedCamera
from pickup_board_piece import move_piece

# --- CONFIGURE HERE ---
PIECE_TYPE  = "king"   # pawn / knight / bishop / rook / queen / king
FROM_SQUARE = "e8"     # starting square (algebraic)
TO_SQUARE   = "g8"     # destination square (algebraic)
REPETITIONS = 5        # back-and-forth cycles (each cycle = 2 manipulations)
# ----------------------


def run_manipulation_cycles(piece_type: str, from_square: str, to_square: str, repetitions: int) -> None:
    """Execute back-and-forth manipulation cycles for a single piece."""
    zed = ZedCamera()
    try:
        for i in range(repetitions):
            cycle = i + 1
            print(f"\n=== Cycle {cycle}/{repetitions}: {from_square} -> {to_square} ===")
            move_piece(piece_type, from_square, to_square, zed)

            print(f"=== Cycle {cycle}/{repetitions}: {to_square} -> {from_square} ===")
            move_piece(piece_type, to_square, from_square, zed)

        print("\nAll cycles complete.")
    finally:
        zed.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeated pick/place manipulation cycles.")
    parser.add_argument("--piece", default=PIECE_TYPE, help="Piece type: pawn/knight/bishop/rook/queen/king")
    parser.add_argument("--from-square", default=FROM_SQUARE, help="Start square (e.g. e2)")
    parser.add_argument("--to-square", default=TO_SQUARE, help="Destination square (e.g. e4)")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS, help="Back-and-forth cycle count")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_manipulation_cycles(args.piece, args.from_square, args.to_square, args.repetitions)


if __name__ == "__main__":
    main()
