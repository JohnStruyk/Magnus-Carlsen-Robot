import random
import time

import chess
import cv2
from xarm.wrapper import XArmAPI

from pickup_board_piece import (
    CHESSBOARD_TAG_FAMILY,
    GRIPPER_LENGTH_M,
    PLAYMAT_TAG_FAMILY,
    ROBOT_IP_DEFAULT,
    build_forward_entry_pose,
    build_graveyard_pose,
    build_vision_from_piece_continuity,
    capture_piece,
    hover_pose,
    move_piece,
    move_to_graveyard_mod180_joints,
    normalize_piece_type,
)
from utils.zed_camera import ZedCamera


def stage_from_graveyard(piece_type: str, zed: ZedCamera, robot_ip: str = ROBOT_IP_DEFAULT) -> None:
    """Travel-only: graveyard hover → forward entry → graveyard (legacy helper for this script)."""
    piece_name = normalize_piece_type(piece_type)
    arm = None
    arm_connected = False
    graveyard_hover_pose = None
    try:
        img = zed.image
        if img is None:
            raise RuntimeError("No image from ZED.")
        K = zed.camera_intrinsic
        vision = build_vision_from_piece_continuity(img, K)
        if vision is None:
            raise RuntimeError(
                f"Vision failed: need tags 0–3 on {PLAYMAT_TAG_FAMILY} and {CHESSBOARD_TAG_FAMILY}."
            )
        t_rb = vision.t_robot_board
        graveyard_hover_pose = build_graveyard_pose(vision.robot_frame_centers, t_rb, piece_name)
        forward_entry_pose = build_forward_entry_pose(vision.robot_frame_centers, graveyard_hover_pose)

        arm = XArmAPI(robot_ip)
        arm.connect()
        arm_connected = True
        arm.motion_enable(enable=True)
        arm.set_tcp_offset([0, 0, GRIPPER_LENGTH_M * 1000.0, 0, 0, 0])
        arm.set_mode(0)
        arm.set_state(0)
        hover_pose(arm, graveyard_hover_pose)
        hover_pose(arm, forward_entry_pose)
        hover_pose(arm, graveyard_hover_pose)
    finally:
        if arm is not None:
            try:
                arm.stop_lite6_gripper()
            except Exception:
                pass
            if arm_connected:
                time.sleep(0.2)
                try:
                    move_to_graveyard_mod180_joints(arm, graveyard_hover_pose)
                except Exception:
                    pass
                try:
                    arm.disconnect()
                except Exception:
                    pass
        cv2.destroyAllWindows()


def execute_robot_move_on_board(board: chess.Board, move: chess.Move, zed: ZedCamera) -> None:
    """Execute one legal move physically using existing pickup logic."""
    move_string = move.uci()
    print(f"[verify] Executing random move: {move_string}")

    if board.is_castling(move):
        king_from_sq = move.from_square
        king_to_sq = move.to_square
        king_piece = board.piece_at(king_from_sq)
        if king_piece is None:
            raise RuntimeError("Castling failed: king piece not found.")

        if board.is_kingside_castling(move):
            rook_from_sq = chess.H1 if board.turn == chess.WHITE else chess.H8
            rook_to_sq = chess.F1 if board.turn == chess.WHITE else chess.F8
        else:
            rook_from_sq = chess.A1 if board.turn == chess.WHITE else chess.A8
            rook_to_sq = chess.D1 if board.turn == chess.WHITE else chess.D8

        rook_piece = board.piece_at(rook_from_sq)
        if rook_piece is None:
            raise RuntimeError("Castling failed: rook piece not found.")

        move_piece(
            king_piece.symbol(),
            chess.square_name(king_from_sq),
            chess.square_name(king_to_sq),
            zed,
        )
        stage_from_graveyard(rook_piece.symbol(), zed)
        move_piece(
            rook_piece.symbol(),
            chess.square_name(rook_from_sq),
            chess.square_name(rook_to_sq),
            zed,
        )
        return

    from_sq = move.from_square
    to_sq = move.to_square
    from_piece = board.piece_at(from_sq)
    to_piece = board.piece_at(to_sq)

    if from_piece is None:
        raise RuntimeError(f"No piece on from-square {chess.square_name(from_sq)}")

    from_symbol = from_piece.symbol()
    from_square = chess.square_name(from_sq)
    to_square = chess.square_name(to_sq)

    if to_piece is not None:
        capture_count = sum(1 for p in board.piece_map().values() if p.color == chess.WHITE)
        capture_count = 16 - capture_count
        capture_piece(from_symbol, to_piece.symbol(), from_square, to_square, zed, capture_count)
    else:
        move_piece(from_symbol, from_square, to_square, zed)


def main() -> None:
    zed = ZedCamera()
    board = chess.Board()

    try:
        while True:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                print("[verify] No legal moves left. Resetting board to start position.")
                board = chess.Board()
                time.sleep(1.0)
                continue

            move = random.choice(legal_moves)
            execute_robot_move_on_board(board, move, zed)
            board.push(move)
            print(f"[verify] Board FEN after move: {board.fen()}")
            time.sleep(0.8)
    finally:
        zed.close()


if __name__ == "__main__":
    main()
