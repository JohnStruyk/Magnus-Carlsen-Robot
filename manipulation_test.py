"""
Manipulation stress test — moves a piece back and forth 5 times (10 total manipulations).
Each call to move_piece handles vision, arm connect/disconnect, and graveyard return internally.

Edit the three constants below to configure the test.
"""

from utils.zed_camera import ZedCamera
from pickup_board_piece import move_piece

# --- CONFIGURE HERE ---
PIECE_TYPE  = "pawn"   # pawn / knight / bishop / rook / queen / king
FROM_SQUARE = "e2"     # starting square (algebraic)
TO_SQUARE   = "e4"     # destination square (algebraic)
REPETITIONS = 5        # back-and-forth cycles (each cycle = 2 manipulations)
# ----------------------


def main():
    zed = ZedCamera()
    try:
        for i in range(REPETITIONS):
            cycle = i + 1
            print(f"\n=== Cycle {cycle}/{REPETITIONS}: {FROM_SQUARE} → {TO_SQUARE} ===")
            move_piece(PIECE_TYPE, FROM_SQUARE, TO_SQUARE, zed)

            print(f"=== Cycle {cycle}/{REPETITIONS}: {TO_SQUARE} → {FROM_SQUARE} ===")
            move_piece(PIECE_TYPE, TO_SQUARE, FROM_SQUARE, zed)

        print("\nAll cycles complete.")
    finally:
        zed.close()


if __name__ == "__main__":
    main()
