"""CLI for one-off ``move_piece`` / ``capture_piece`` runs (keeps ``pickup_board_piece`` import-light)."""

from __future__ import annotations

import argparse

from pickup_board_piece import ROBOT_IP_DEFAULT, capture_piece, move_piece
from utils.zed_camera import ZedCamera


def main() -> None:
    p = argparse.ArgumentParser(description="Pick and place a chess piece (ZED + AprilTags + Lite6).")
    p.add_argument("--piece-type", required=True, help="pawn/knight/bishop/rook/queen/king or P/N/B/R/Q/K")
    p.add_argument("--from-square", required=True, help="Source square, e.g. e5")
    p.add_argument("--to-square", required=True, help="Destination square, e.g. e4")
    p.add_argument("--captured-piece-type", required=False, help="For captures: victim piece type")
    p.add_argument("--robot-ip", default=ROBOT_IP_DEFAULT)
    a = p.parse_args()

    zed = ZedCamera()
    try:
        if a.captured_piece_type:
            capture_piece(
                a.piece_type,
                a.captured_piece_type,
                a.from_square,
                a.to_square,
                zed,
                capture_count=1,
                robot_ip=a.robot_ip,
            )
        else:
            move_piece(a.piece_type, a.from_square, a.to_square, zed, robot_ip=a.robot_ip)
    finally:
        zed.close()


if __name__ == "__main__":
    main()
