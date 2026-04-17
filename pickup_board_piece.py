"""
Pick / place using RRC-style geometry:
  - Playmat (tags 0–3): checkpoint0.get_transform_camera_robot -> T_cam_robot
  - Board: piece_continuity BOARD_CONFIG geometry with tag IDs BOARD_TAG_IDS (default 4–7)
    so corners are not confused with playmat tags in one image.

Physical setup: print AprilTags on the board corners as IDs 4,5,6,7 (same layout as
BOARD_CONFIG corners 0–3). Use ZED left-camera intrinsics from ZedCamera.
"""
import argparse
import time

import cv2
import numpy as np
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from checkpoint0 import get_transform_camera_robot
from piece_continuity import (
    BOARD_CONFIG,
    detect_pieces,
    get_4x4_transform,
    get_board_centers_local,
    get_warped,
)
from utils.zed_camera import ZedCamera


# RRC-style geometry: playmat gives T_cam_robot via checkpoint0; board gives T_cam_board via piece_continuity.
# Board tags must NOT share IDs 0–3 with the playmat or PnP will mix corners. Default: board corners use IDs 4–7
# mapped to the same physical layout as BOARD_CONFIG tag_centers 0–3.
BOARD_TAG_IDS = [4, 5, 6, 7]

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


def _board_config_for_pnp():
    """
    Same board geometry as piece_continuity BOARD_CONFIG, but tag IDs 4–7 so they never
    collide with playmat tags 0–3 in the same image (required for correct PnP).
    """
    c = BOARD_CONFIG["tag_centers"]
    return {
        **BOARD_CONFIG,
        "tag_ids": BOARD_TAG_IDS,
        "tag_centers": {
            BOARD_TAG_IDS[0]: c[0],
            BOARD_TAG_IDS[1]: c[1],
            BOARD_TAG_IDS[2]: c[2],
            BOARD_TAG_IDS[3]: c[3],
        },
    }


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
      - Playmat / robot frame: checkpoint0.get_transform_camera_robot -> T_cam_robot
      - Board frame: piece_continuity.get_4x4_transform(BOARD) -> T_cam_board
      - Point in robot base: p_robot = inv(T_cam_robot) @ T_cam_board @ p_board
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if img.shape[-1] == 4 else cv2.COLOR_BGR2GRAY)
    detector = Detector(families="tag36h11")
    tags = detector.detect(gray)

    # Playmat: same PnP as checkpoint0 / RRC (tags 0–3 on mat only).
    t_cam_robot = get_transform_camera_robot(img, camera_intrinsic)
    if t_cam_robot is None:
        return None

    board_cfg = _board_config_for_pnp()
    t_board_to_cam, b_rvec, b_tvec = get_4x4_transform(
        tags, board_cfg, camera_intrinsic, strict=True
    )
    if t_board_to_cam is None:
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
    }


def show_preview(raw_img, warped, from_row, from_col, to_row, to_col):
    """
    Display warped board with source/destination overlays.
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

    cv2.namedWindow("Raw camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Warped board", cv2.WINDOW_NORMAL)
    cv2.imshow("Raw camera", raw_img)
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
                "Vision failed: need playmat tags 0–3 (checkpoint0) and board tags "
                f"{BOARD_TAG_IDS} with geometry from piece_continuity BOARD_CONFIG."
            )

        warped = vision["warped"]
        board_state = vision["board_state"]
        robot_frame_centers = vision["robot_frame_centers"]
        should_execute = True
        if preview:
            print("[pickup] Showing preview window (press 'k' to execute)...")
            should_execute = show_preview(img, warped, from_row, from_col, to_row, to_col)

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
    parser.add_argument("--preview", action="store_true", help="Show board preview, execute on 'k'.")
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
