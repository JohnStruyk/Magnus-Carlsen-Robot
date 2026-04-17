import argparse
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from checkpoint0 import (
    CHESSBOARD_TAG_FAMILY,
    PLAYMAT_TAG_FAMILY,
    TAG_SIZE,
    best_tag_per_id_0_3,
    detect_playmat_and_chessboard_tags,
    draw_dual_family_tag_overlays,
    get_transform_camera_robot_from_tags,
    resize_for_preview,
    to_bgr_display,
)
from utils.vis_utils import draw_pose_axes
from piece_continuity import (
    BOARD_CONFIG,
    detect_pieces,
    get_4x4_transform,
    get_board_centers_local,
    get_warped,
)
from utils.zed_camera import ZedCamera


# Playmat vs chessboard use different AprilTag families; both use ids 0–3 (see checkpoint0).

ROBOT_IP_DEFAULT = "192.168.1.159"
SAFE_Z = 0.22
GRASP_Z_OFFSET = 0.0001
LIFT_Z_DELTA = 0.06
PLACE_Z_OFFSET = 0.002
TOOL_ROLL_DEG = 180.0
TOOL_PITCH_DEG = 0.0
GRIPPER_LENGTH_M = 0.067
# Very slow for safety / tuning.
ARM_SPEED_TRAVEL_MM_S = 80
ARM_SPEED_DESCEND_MM_S = 30
GRIPPER_SETTLE_AFTER_OPEN_S = 0.40
GRIPPER_SETTLE_AFTER_CLOSE_S = 0.55
GRIPPER_SETTLE_AFTER_RELEASE_S = 0.40
GRASP_DWELL_BEFORE_CLOSE_S = 0.25

def move_to_pose(arm: XArmAPI, t_robot_target: np.ndarray, z_offset_m: float, descend_speed: int):
    xyz = t_robot_target[:3, 3]
    x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
    safe_z_mm = SAFE_Z * 1000.0
    target_z_mm = z_mm + (z_offset_m * 1000.0)
    lift_z_mm = max(safe_z_mm, target_z_mm + (LIFT_Z_DELTA * 1000.0))

    r = Rotation.from_matrix(t_robot_target[:3, :3])
    _, _, yaw_deg = r.as_euler("xyz", degrees=True)

    arm.set_position(
        x_mm, y_mm, safe_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, yaw_deg,
        speed=ARM_SPEED_TRAVEL_MM_S, is_radian=False, wait=True
    )
    arm.set_position(
        x_mm, y_mm, target_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, yaw_deg,
        speed=descend_speed, is_radian=False, wait=True
    )
    return x_mm, y_mm, lift_z_mm, yaw_deg


def pickup_pose(arm: XArmAPI, t_robot_target: np.ndarray):
    arm.open_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_AFTER_OPEN_S)
    x_mm, y_mm, lift_z_mm, yaw_deg = move_to_pose(
        arm, t_robot_target, GRASP_Z_OFFSET, ARM_SPEED_DESCEND_MM_S
    )
    time.sleep(GRASP_DWELL_BEFORE_CLOSE_S)
    arm.close_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_AFTER_CLOSE_S)
    arm.set_position(
        x_mm, y_mm, lift_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, yaw_deg,
        speed=ARM_SPEED_TRAVEL_MM_S, is_radian=False, wait=True
    )


def place_pose(arm: XArmAPI, t_robot_target: np.ndarray):
    x_mm, y_mm, lift_z_mm, yaw_deg = move_to_pose(
        arm, t_robot_target, PLACE_Z_OFFSET, ARM_SPEED_DESCEND_MM_S
    )
    arm.open_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_AFTER_RELEASE_S)
    arm.set_position(
        x_mm, y_mm, lift_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, yaw_deg,
        speed=ARM_SPEED_TRAVEL_MM_S, is_radian=False, wait=True
    )


def algebraic_to_row_col(square: str):
    if len(square) != 2 or square[0].lower() not in "abcdefgh" or square[1] not in "12345678":
        raise ValueError(f"Invalid square '{square}'. Use algebraic notation like 'b3'.")
    file_idx = ord(square[0].lower()) - ord("a")
    rank = int(square[1])
    row = 8 - rank
    col = file_idx
    return row, col


def normalize_piece_type(piece_type):
    if isinstance(piece_type, int):
        raise ValueError("Use chess piece type (pawn/knight/bishop/rook/queen/king), not int.")
    value = str(piece_type).strip().lower()
    mapping = {
        "p": "pawn",
        "pawn": "pawn",
        "n": "knight",
        "knight": "knight",
        "b": "bishop",
        "bishop": "bishop",
        "r": "rook",
        "rook": "rook",
        "q": "queen",
        "queen": "queen",
        "k": "king",
        "king": "king",
    }
    if value not in mapping:
        raise ValueError("piece_type must be one of {pawn, knight, bishop, rook, queen, king} (or P/N/B/R/Q/K).")
    return mapping[value]


def square_to_robot_pose(robot_frame_centers, row, col, t_robot_board):
    """Position from mapped centers; orientation from board frame in robot base (RRC-style)."""
    idx = row * 8 + col
    p_robot = robot_frame_centers[idx]
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = t_robot_board[:3, :3].astype(np.float32)
    pose[:3, 3] = np.array(p_robot[:3], dtype=np.float32)
    return pose


def build_vision_from_piece_continuity(img, camera_intrinsic):
    """
    RRC geometry (see Real-Robot-Challenge/checkpoint1.py):
      - Playmat / robot frame: checkpoint0 (PLAYMAT_TAG_FAMILY, ids 0–3) -> T_cam_robot
      - Board frame: piece_continuity.get_4x4_transform(CHESSBOARD_TAG_FAMILY, ids 0–3) -> T_cam_board
      - Point in robot base: p_robot = inv(T_cam_robot) @ T_cam_board @ p_board
    """
    _, playmat_raw, chess_raw = detect_playmat_and_chessboard_tags(img)
    playmat_tags, pm_ok = best_tag_per_id_0_3(playmat_raw)
    board_tags, ch_ok = best_tag_per_id_0_3(chess_raw)
    split_msg = (
        f"dual-family playmat {PLAYMAT_TAG_FAMILY} n={len(playmat_raw)} (ok 4 corners: {pm_ok}), "
        f"chess {CHESSBOARD_TAG_FAMILY} n={len(chess_raw)} (ok 4 corners: {ch_ok})"
    )
    if not pm_ok:
        print(f"[pickup] Need four playmat tags ids 0-3 in {PLAYMAT_TAG_FAMILY}. {split_msg}")
        print(f"[pickup] Raw playmat ids: {sorted(set(int(t.tag_id) for t in playmat_raw))}")
        return None
    if not ch_ok:
        print(f"[pickup] Need four chessboard tags ids 0-3 in {CHESSBOARD_TAG_FAMILY}. {split_msg}")
        print(f"[pickup] Raw chess ids: {sorted(set(int(t.tag_id) for t in chess_raw))}")
        return None
    print(f"[pickup] {split_msg}")

    t_cam_robot = get_transform_camera_robot_from_tags(playmat_tags, camera_intrinsic)
    if t_cam_robot is None:
        return None

    t_board_to_cam, b_rvec, b_tvec = get_4x4_transform(
        board_tags, BOARD_CONFIG, camera_intrinsic, strict=True
    )
    if t_board_to_cam is None:
        print("[pickup] Board PnP failed (need 4 board corners visible).")
        return None

    # checkpoint1: t_robot_cam = inv(camera_pose) with camera_pose = T_cam_robot
    t_robot_cam = np.linalg.inv(t_cam_robot)
    t_robot_board = t_robot_cam @ t_board_to_cam

    local_centers = get_board_centers_local(BOARD_CONFIG)
    robot_frame_centers = {}
    for i, p_local in enumerate(local_centers):
        p_robot = t_robot_cam @ (t_board_to_cam @ p_local)
        robot_frame_centers[i] = p_robot[:3].tolist()

    warped, _ = get_warped(img, b_rvec, b_tvec, camera_intrinsic, square_px=100)
    board_state = detect_pieces(warped, square_px=100)
    return {
        "warped": warped,
        "board_state": board_state,
        "robot_frame_centers": robot_frame_centers,
        "t_robot_board": t_robot_board,
        "playmat_tags": playmat_raw,
        "chessboard_tags": chess_raw,
        "tags": list(playmat_raw) + list(chess_raw),
        "t_cam_robot": t_cam_robot,
        "split_msg": split_msg,
    }


def show_preview(
    raw_img,
    warped,
    camera_intrinsic,
    from_row,
    from_col,
    to_row,
    to_col,
    vision_meta=None,
):
    """
    Display checkpoint0-style AprilTag overlay (all tags + playmat pose axes) and warped board
    with source/destination overlays. Same key as checkpoint0: press 'k' to confirm execute.
    """
    preview_img = warped.copy()
    square_px = warped.shape[0] // 8

    def square_rect(row, col):
        x0 = col * square_px
        y0 = row * square_px
        x1 = x0 + square_px
        y1 = y0 + square_px
        return x0, y0, x1, y1

    fx0, fy0, fx1, fy1 = square_rect(from_row, from_col)
    tx0, ty0, tx1, ty1 = square_rect(to_row, to_col)
    cv2.rectangle(preview_img, (fx0, fy0), (fx1, fy1), (0, 255, 255), 3)
    cv2.rectangle(preview_img, (tx0, ty0), (tx1, ty1), (255, 0, 0), 3)
    cv2.putText(preview_img, "FROM", (fx0 + 4, fy0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(preview_img, "TO", (tx0 + 4, ty0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Downscale large warped views so preview is usable on screen.
    max_preview_dim = 900
    h, w = preview_img.shape[:2]
    scale = min(max_preview_dim / float(max(h, w)), 1.0)
    if scale < 1.0:
        preview_img = cv2.resize(
            preview_img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    if vision_meta and vision_meta.get("playmat_tags") is not None:
        tag_vis = to_bgr_display(raw_img)
        if tag_vis is not None:
            tag_vis = tag_vis.copy()
            draw_dual_family_tag_overlays(
                tag_vis,
                vision_meta["playmat_tags"],
                vision_meta["chessboard_tags"],
            )
            tcr = vision_meta.get("t_cam_robot")
            if tcr is not None:
                draw_pose_axes(tag_vis, camera_intrinsic, tcr, size=TAG_SIZE)
            status = vision_meta.get("split_msg", "")
            line = f"pickup: {status} | chessboard PnP OK"
            for dy, color, th in ((2, (255, 255, 255), 2), (0, (0, 200, 0), 1)):
                cv2.putText(
                    tag_vis,
                    line,
                    (12, 86 + dy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    th,
                    cv2.LINE_AA,
                )
            cv2.namedWindow("pickup: AprilTag preview", cv2.WINDOW_NORMAL)
            cv2.imshow("pickup: AprilTag preview", resize_for_preview(tag_vis))

    cv2.namedWindow("Warped board", cv2.WINDOW_NORMAL)
    cv2.imshow("Warped board", preview_img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key == ord("k")


def move_piece(piece_type, from_square, to_square, robot_ip=ROBOT_IP_DEFAULT, preview=False):
    piece_name = normalize_piece_type(piece_type)
    from_row, from_col = algebraic_to_row_col(from_square)
    to_row, to_col = algebraic_to_row_col(to_square)

    zed = ZedCamera()
    arm = None

    try:
        print("[pickup] Capturing camera image...")
        img = zed.image
        if img is None:
            raise RuntimeError("No image from ZED.")

        print("[pickup] Running vision (RRC playmat + piece_continuity board)...")
        camera_intrinsic = zed.camera_intrinsic
        vision = build_vision_from_piece_continuity(img, camera_intrinsic)
        if vision is None:
            raise RuntimeError(
                f"Vision failed: need four corners ids 0–3 in {PLAYMAT_TAG_FAMILY} (playmat) "
                f"and four in {CHESSBOARD_TAG_FAMILY} (chessboard). See console lines above."
            )

        warped = vision["warped"]
        board_state = vision["board_state"]
        robot_frame_centers = vision["robot_frame_centers"]
        should_execute = True
        if preview:
            print(
                "[pickup] Showing preview (AprilTag + warped board; press 'k' to execute)..."
            )
            should_execute = show_preview(
                img,
                warped,
                camera_intrinsic,
                from_row,
                from_col,
                to_row,
                to_col,
                vision_meta={
                    "playmat_tags": vision["playmat_tags"],
                    "chessboard_tags": vision["chessboard_tags"],
                    "t_cam_robot": vision["t_cam_robot"],
                    "split_msg": vision["split_msg"],
                },
            )

        from_detected = int(board_state[from_row, from_col])
        if from_detected == 0:
            raise RuntimeError(
                f"Source square {from_square} appears empty. "
                f"Detected value={from_detected}."
            )

        t_rb = vision["t_robot_board"]
        from_pose = square_to_robot_pose(robot_frame_centers, from_row, from_col, t_rb)
        to_pose = square_to_robot_pose(robot_frame_centers, to_row, to_col, t_rb)

        print(f"Moving {piece_name} from {from_square} to {to_square}")
        print(f"From xyz (m): {from_pose[:3, 3].tolist()}")
        print(f"To xyz (m): {to_pose[:3, 3].tolist()}")

        if should_execute:
            print("[pickup] Connecting to arm...")
            arm = XArmAPI(robot_ip)
            arm.connect()
            arm.motion_enable(enable=True)
            arm.set_tcp_offset([0, 0, GRIPPER_LENGTH_M * 1000.0, 0, 0, 0])
            arm.set_mode(0)
            arm.set_state(0)
            arm.move_gohome(wait=True)
            time.sleep(0.5)
            print("[pickup] Executing pick/place motion...")
            pickup_pose(arm, from_pose)
            place_pose(arm, to_pose)
        else:
            print("Cancelled (press 'k' in preview to execute).")
    finally:
        if arm is not None:
            arm.stop_lite6_gripper()
            arm.move_gohome(wait=True)
            time.sleep(0.5)
            arm.disconnect()
        zed.close()
        cv2.destroyAllWindows()


def move_piece_three_params(piece_type, from_square, to_square):
    """
    Three-parameter API requested by project code.
    """
    move_piece(piece_type, from_square, to_square)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Move a piece using Magnus camera functions."
    )
    parser.add_argument("--piece-type", required=True, help="Chess piece: pawn/knight/bishop/rook/queen/king or P/N/B/R/Q/K.")
    parser.add_argument("--from-square", required=True, help="Source square in algebraic notation, e.g. b3.")
    parser.add_argument("--to-square", required=True, help="Destination square in algebraic notation, e.g. c4.")
    parser.add_argument("--robot-ip", type=str, default=ROBOT_IP_DEFAULT)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show AprilTag overlay (checkpoint0-style) + warped board; press 'k' to execute.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    move_piece(
        args.piece_type,
        args.from_square,
        args.to_square,
        robot_ip=args.robot_ip,
        preview=args.preview,
    )


if __name__ == "__main__":
    main()
