"""Lite6 pick/place using one camera frame -> PickVision (tags + warp + occupancy)."""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from utils.calibrate_tags import (
    CHESSBOARD_TAG_FAMILY,
    PLAYMAT_TAG_FAMILY,
    ROBOT_IP_DEFAULT,
    best_tag_per_id_0_3,
    detect_playmat_and_chessboard_tags,
    get_transform_camera_robot_from_tags,
)
from piece_continuity import (
    BOARD_CONFIG,
    detect_pieces,
    get_4x4_transform,
    get_board_centers_local,
    get_warped,
)
from utils.zed_camera import ZedCamera

# Tune these when the rig changes (height, gripper, board tilt).

_VISION_TAG_ERR = (
    f"Vision failed: need four tags ids 0–3 on {PLAYMAT_TAG_FAMILY} (playmat) "
    f"and four on {CHESSBOARD_TAG_FAMILY} (chessboard)."
)

BOARD_TAG_EDGE_M = BOARD_CONFIG["tag_size"]  # Same board sheet as vision, explicit for pickup PnP.
HAND_EYE_XYZ_BIAS_M = np.zeros(3, dtype=np.float64)  # Optional mm skew cam-hand calibration (often zeros).
USE_HARDCODED_SQUARE_CENTERS = False

SAFE_Z = 0.14
GRASP_Z_OFFSET = 0.0001
LIFT_Z_DELTA = 0.06
PLACE_Z_OFFSET = 0.002
ROW1_Z_ADJUST_M = -0.004
ROW1_Z_ADJUST_RADIUS_ROWS = 2.0
LOWER_ROWS_CENTER_SHIFT_M = 0.0
LOWER_ROWS_START_ROW = 5
ROW_DEPTH_CORRECTION_MAX_M = 0.005
MIN_TOOL_Z_M = 0.04
TOOL_ROLL_DEG = 180.0
TOOL_PITCH_DEG = 0.0
MAX_ABS_TOOL_YAW_DEG = 120.0
JOINT_UNWIND_SPEED_DEG_S = 20
JOINT_UNWIND_ACCEL_DEG_S2 = 100
JOINT_RETURN_POSE_DEG = [90.0, 0.0, 0.0, 0.0, 0.0, 0.0]
GRIPPER_LENGTH_M = 0.067
ARM_SPEED_TRAVEL_MM_S = 80
ARM_SPEED_DESCEND_MM_S = 30
GRIPPER_SETTLE_AFTER_OPEN_S = 0.40
GRIPPER_SETTLE_AFTER_CLOSE_S = 0.55
GRIPPER_SETTLE_AFTER_RELEASE_S = 0.40
GRASP_DWELL_BEFORE_CLOSE_S = 0.25
FORWARD_ENTRY_BOARD_FRACTION = 0.7
MAX_FORWARD_ENTRY_STEP_M = 0.12
GRAVEYARD_ANCHOR_ROW = 0
GRAVEYARD_ANCHOR_COL = 7
GRAVEYARD_X_SHIFT_M = -0.15

PROMOTION_SOURCE_X_M: Optional[float] = 0.221
PROMOTION_SOURCE_Y_M: Optional[float] = -0.2942
PROMOTION_SOURCE_Z_M: Optional[float] = 0.1511
PROMOTION_SOURCE_YAW_DEG = -32.8
PROMOTION_SOURCE_GRASP_TARGET_Z_M = 0.075
PROMOTION_PAWN_DISCARD_X_OFFSET_M = 0.05

GRASP_Z_OFFSET_PAWN_M = 0.045
GRASP_Z_OFFSET_KNIGHT_M = 0.05
GRASP_Z_OFFSET_BISHOP_M = 0.06
GRASP_Z_OFFSET_ROOK_M = 0.045
GRASP_Z_OFFSET_QUEEN_M = 0.075
GRASP_Z_OFFSET_KING_M = 0.085

PIECE_GRASP_Z_OFFSET_M = {
    "pawn": GRASP_Z_OFFSET_PAWN_M,
    "knight": GRASP_Z_OFFSET_KNIGHT_M,
    "bishop": GRASP_Z_OFFSET_BISHOP_M,
    "rook": GRASP_Z_OFFSET_ROOK_M,
    "queen": GRASP_Z_OFFSET_QUEEN_M,
    "king": GRASP_Z_OFFSET_KING_M,
}


@dataclass(frozen=True)
class PickVision:
    """Occupancy grid + 64 square centers in base frame + board orientation."""

    board_state: np.ndarray
    robot_frame_centers: Dict[int, List[float]]
    t_robot_board: np.ndarray


def algebraic_to_row_col(square: str) -> Tuple[int, int]:
    """Algebra like e4 -> row/col in the warped board image."""
    if len(square) != 2 or square[0].lower() not in "abcdefgh" or square[1] not in "12345678":
        raise ValueError(f"Invalid square '{square}'. Use algebraic notation like 'b3'.")
    col = ord(square[0].lower()) - ord("a")
    row = 8 - int(square[1])
    return row, col


def normalize_piece_type(piece_type: str) -> str:
    """Accept P/N/... or full names. Returns internal keys like pawn/knight."""
    if isinstance(piece_type, int):
        raise ValueError("Use chess piece type (pawn/knight/...), not int.")
    key = str(piece_type).strip().lower()
    m = {
        "p": "pawn", "pawn": "pawn",
        "n": "knight", "knight": "knight",
        "b": "bishop", "bishop": "bishop",
        "r": "rook", "rook": "rook",
        "q": "queen", "queen": "queen",
        "k": "king", "king": "king",
    }
    if key not in m:
        raise ValueError("piece_type must be pawn/knight/bishop/rook/queen/king (or P/N/B/R/Q/K).")
    return m[key]


def board_config_for_pickup() -> dict:
    """BOARD_CONFIG plus tag edge length used here for PnP."""
    return {**BOARD_CONFIG, "tag_size": float(BOARD_TAG_EDGE_M)}


def warp_cell_center_to_robot_xyz(
    row: int,
    col: int,
    square_px: int,
    H_warp_to_img: np.ndarray,
    K: np.ndarray,
    t_board_to_cam: np.ndarray,
    t_robot_cam: np.ndarray,
) -> Optional[np.ndarray]:
    """Warp cell -> pixel ray -> board plane -> base XYZ. None -> caller falls back to metric grid."""
    # Cell center in warped-board pixels (v scaled by 0.97 to match slight warp stretch).
    u_w = float(col * square_px + square_px * 0.5)
    v_w = float(row * square_px * 0.97 + square_px * 0.5)
    ph = H_warp_to_img @ np.array([u_w, v_w, 1.0], dtype=np.float64)  # homogeneous raw-pixel coords
    if abs(ph[2]) < 1e-12:
        return None
    u, v = ph[0] / ph[2], ph[1] / ph[2]
    d = np.linalg.inv(K) @ np.array([u, v, 1.0], dtype=np.float64)  # unit ray direction in cam frame
    R_bc = t_board_to_cam[:3, :3].astype(np.float64)
    t_bc = t_board_to_cam[:3, 3].astype(np.float64)
    n = R_bc[:, 2]  # board plane normal in camera frame
    denom = float(np.dot(n, d))
    if abs(denom) < 1e-10:
        return None
    alpha = float(np.dot(n, t_bc) / denom)  # ray parameter to plane intersection
    p_cam = alpha * d  # 3D contact point in camera frame
    return (t_robot_cam @ np.append(p_cam, 1.0))[:3].astype(np.float64)


def compute_robot_frame_centers(
    board_cfg: dict,
    square_px: int,
    H_warp_to_img: np.ndarray,
    K: np.ndarray,
    t_board_to_cam: np.ndarray,
    t_robot_cam: np.ndarray,
) -> Dict[int, List[float]]:
    """Per-square base XYZ. Prefer homography ray, else metric chain via ``get_board_centers_local``."""
    local = get_board_centers_local(board_cfg)
    out: Dict[int, List[float]] = {}
    idx = 0
    for r in range(8):
        for c in range(8):
            p_fb = (t_robot_cam @ (t_board_to_cam @ local[idx]))[:3]
            if USE_HARDCODED_SQUARE_CENTERS:
                out[idx] = p_fb.astype(np.float64).tolist()
            else:
                p = warp_cell_center_to_robot_xyz(
                    r, c, square_px, H_warp_to_img, K, t_board_to_cam, t_robot_cam
                )
                # p_fb: fallback XYZ from tag geometry only (no homography ray).
                out[idx] = (p if p is not None else p_fb.astype(np.float64)).tolist()
            idx += 1
    return out


def piece_grasp_vertical_offset_m(piece_name: str) -> float:
    if piece_name not in PIECE_GRASP_Z_OFFSET_M:
        raise KeyError(f"No grasp Z offset for piece '{piece_name}'. Add it to PIECE_GRASP_Z_OFFSET_M.")
    return float(PIECE_GRASP_Z_OFFSET_M[piece_name])


def square_to_robot_pose(
    robot_frame_centers: Dict[int, List[float]],
    row: int,
    col: int,
    t_robot_board: np.ndarray,
    piece_name: Optional[str] = None,
) -> np.ndarray:
    """Square center in base plus board rotation. ``piece_name`` adds grasp height on Z only."""
    idx = row * 8 + col
    t = np.asarray(robot_frame_centers[idx][:3], dtype=np.float64) + HAND_EYE_XYZ_BIAS_M
    # Linear depth correction: 0 mm at row 0 (near base), ROW_DEPTH_CORRECTION_MAX_M at row 7.
    depth_weight = float(row) / 7.0
    depth_correction = ROW_DEPTH_CORRECTION_MAX_M * depth_weight
    board_x_axis_in_robot = t_robot_board[:3, 0].astype(np.float64)
    t = t.copy()
    t[:3] += board_x_axis_in_robot * depth_correction
    # Blend a lateral correction for lower rows to improve centering near the arm.
    if row >= LOWER_ROWS_START_ROW and LOWER_ROWS_CENTER_SHIFT_M != 0.0:
        denom = max(1, 7 - LOWER_ROWS_START_ROW)
        w = float(row - LOWER_ROWS_START_ROW) / float(denom)
        board_y_axis_in_robot = t_robot_board[:3, 1].astype(np.float64)
        t = t.copy()
        t[:3] += board_y_axis_in_robot * (LOWER_ROWS_CENTER_SHIFT_M * w)
    # Smoothly taper row-1 Z compensation across nearby rows.
    # Row 1 gets full ROW1_Z_ADJUST_M. Nearby ranks get less via ``weight``.
    dist_from_row1 = abs(float(row) - 1.0)
    weight = max(0.0, 1.0 - (dist_from_row1 / ROW1_Z_ADJUST_RADIUS_ROWS))
    row_z_adjust = ROW1_Z_ADJUST_M * weight
    if row_z_adjust != 0.0:
        t = t.copy()
        t[2] += row_z_adjust
    if piece_name is not None:
        t = t.copy()
        t[2] += piece_grasp_vertical_offset_m(piece_name)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = t_robot_board[:3, :3].astype(np.float32)
    pose[:3, 3] = t.astype(np.float32)
    return pose


def build_vision_from_piece_continuity(
    img: np.ndarray, camera_intrinsic: np.ndarray
) -> Optional[PickVision]:
    """Tags -> cam<->robot + board pose -> warped frame -> centers + ``detect_pieces`` grid. None if tags bad."""
    # Gray channel unused here (detectors run inside helper).
    _, playmat_raw, chess_raw = detect_playmat_and_chessboard_tags(img)
    playmat_tags, pm_ok = best_tag_per_id_0_3(playmat_raw)
    board_tags, ch_ok = best_tag_per_id_0_3(chess_raw)
    if not pm_ok:
        return None
    if not ch_ok:
        return None

    board_cfg = board_config_for_pickup()
    if not np.isfinite(camera_intrinsic).all():
        raise RuntimeError("[pickup] camera_intrinsic has non-finite values.")

    t_cam_robot = get_transform_camera_robot_from_tags(playmat_tags, camera_intrinsic)
    if t_cam_robot is None:
        return None
    if not np.isfinite(t_cam_robot).all():
        raise RuntimeError("[pickup] t_cam_robot has non-finite values.")

    t_board_to_cam, b_rvec, b_tvec = get_4x4_transform(
        board_tags, board_cfg, camera_intrinsic, strict=True
    )
    if t_board_to_cam is None:
        return None
    if not np.isfinite(t_board_to_cam).all():
        raise RuntimeError("[pickup] t_board_to_cam has non-finite values.")
    if not np.isfinite(b_rvec).all() or not np.isfinite(b_tvec).all():
        raise RuntimeError("[pickup] board rvec/tvec has non-finite values.")

    t_robot_cam = np.linalg.inv(t_cam_robot)
    t_robot_board = t_robot_cam @ t_board_to_cam  # board frame expressed in robot base
    if not np.isfinite(t_robot_board).all():
        raise RuntimeError("[pickup] t_robot_board has non-finite values.")

    square_px = 100  # pixels per square edge in ``get_warped`` / ``detect_pieces``
    warped, _, H_img_to_warp = get_warped(img, b_rvec, b_tvec, camera_intrinsic, square_px=square_px)
    if warped is None or H_img_to_warp is None:
        raise RuntimeError("[pickup] get_warped returned None.")
    if not np.isfinite(H_img_to_warp).all():
        raise RuntimeError("[pickup] H_img_to_warp has non-finite values.")
    H_warp_to_img = np.linalg.inv(H_img_to_warp)  # maps normalized warp coords back to raw image
    if not np.isfinite(H_warp_to_img).all():
        raise RuntimeError("[pickup] H_warp_to_img has non-finite values.")

    centers = compute_robot_frame_centers(
        board_cfg, square_px, H_warp_to_img, camera_intrinsic, t_board_to_cam, t_robot_cam
    )
    if len(centers) != 64:
        raise RuntimeError(f"[pickup] Expected 64 board centers, got {len(centers)}.")
    if not np.isfinite(np.array(list(centers.values()), dtype=np.float64)).all():
        raise RuntimeError("[pickup] robot_frame_centers contains non-finite values.")

    board_state = detect_pieces(warped, square_px=square_px)
    if board_state is None:
        raise RuntimeError("[pickup] detect_pieces returned None.")
    if not np.isfinite(board_state).all():
        raise RuntimeError("[pickup] board_state has non-finite values.")

    return PickVision(
        board_state=board_state,
        robot_frame_centers=centers,
        t_robot_board=t_robot_board,
    )


def move_to_pose(
    arm: XArmAPI,
    t_robot_target: np.ndarray,
    z_offset_m: float,
    descend_speed: int,
    piece_name: Optional[str] = None,
) -> Tuple[float, float, float, float]:
    """Hover at safe Z, plunge to piece, return (x,y,lift_z,yaw) for the retract move."""
    xyz = t_robot_target[:3, 3]
    x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
    safe_z_mm = max(SAFE_Z * 1000.0, MIN_TOOL_Z_M * 1000.0)
    target_z_mm = z_mm + z_offset_m * 1000.0
    z_floor_mm = MIN_TOOL_Z_M * 1000.0
    if target_z_mm < z_floor_mm:
        target_z_mm = z_floor_mm
    lift_z_mm = max(safe_z_mm, target_z_mm + LIFT_Z_DELTA * 1000.0)  # retract height after grasp/release
    _, _, yaw_deg = Rotation.from_matrix(t_robot_target[:3, :3]).as_euler("xyz", degrees=True)
    base_yaw_deg = ((yaw_deg + 180.0) % 360.0) - 180.0  # tool yaw nearest [-180,180]

    if piece_name == "knight":
        # Keep knight gripper orientation truly orthogonal to board yaw.
        # Prefer +90, fallback to -90 if +90 exceeds safety bounds.
        knight_plus = ((base_yaw_deg + 90.0 + 180.0) % 360.0) - 180.0
        knight_minus = ((base_yaw_deg - 90.0 + 180.0) % 360.0) - 180.0
        if abs(knight_plus) <= MAX_ABS_TOOL_YAW_DEG:
            yaw_deg = knight_plus
        elif abs(knight_minus) <= MAX_ABS_TOOL_YAW_DEG:
            yaw_deg = knight_minus
        else:
            # Last resort: clip the preferred orthogonal angle.
            yaw_deg = max(-MAX_ABS_TOOL_YAW_DEG, min(MAX_ABS_TOOL_YAW_DEG, knight_plus))
    else:
        # Normalize then clamp yaw so wrist joints do not chase extreme rotations.
        yaw_deg = max(-MAX_ABS_TOOL_YAW_DEG, min(MAX_ABS_TOOL_YAW_DEG, base_yaw_deg))
    if not np.isfinite([x_mm, y_mm, safe_z_mm, target_z_mm, yaw_deg]).all():
        raise RuntimeError(
            f"[pickup] Non-finite pose for move_to_pose: "
            f"x={x_mm}, y={y_mm}, safe_z={safe_z_mm}, target_z={target_z_mm}, yaw={yaw_deg}"
        )

    arm.set_position(
        x_mm, y_mm, safe_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, yaw_deg,
        speed=ARM_SPEED_TRAVEL_MM_S, is_radian=False, wait=True,
    )
    arm.set_position(
        x_mm, y_mm, target_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, yaw_deg,
        speed=descend_speed, is_radian=False, wait=True,
    )
    return x_mm, y_mm, lift_z_mm, yaw_deg


def move_to_graveyard_mod180_joints(arm: XArmAPI, graveyard_hover_pose: Optional[np.ndarray]) -> None:
    # Park: optional hover then a known joint config (avoids wrist wrap on the way home).
    if graveyard_hover_pose is not None:
        hover_pose(arm, graveyard_hover_pose)

    arm.set_servo_angle(
        angle=JOINT_RETURN_POSE_DEG,
        speed=JOINT_UNWIND_SPEED_DEG_S,
        mvacc=JOINT_UNWIND_ACCEL_DEG_S2,
        is_radian=False,
        wait=True,
    )


def pickup_pose(arm: XArmAPI, t_robot_target: np.ndarray, piece_name: Optional[str] = None) -> None:
    """Open gripper, descend to grasp height, close, lift slightly."""
    arm.open_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_AFTER_OPEN_S)
    x_mm, y_mm, lift_z_mm, yaw_deg = move_to_pose(
        arm, t_robot_target, GRASP_Z_OFFSET, ARM_SPEED_DESCEND_MM_S, piece_name=piece_name
    )
    time.sleep(GRASP_DWELL_BEFORE_CLOSE_S)
    arm.close_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_AFTER_CLOSE_S)
    arm.set_position(
        x_mm, y_mm, lift_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, yaw_deg,
        speed=ARM_SPEED_TRAVEL_MM_S, is_radian=False, wait=True,
    )


def place_pose(arm: XArmAPI, t_robot_target: np.ndarray, piece_name: Optional[str] = None) -> None:
    """Descend to place height, open gripper, retract to safe Z."""
    x_mm, y_mm, lift_z_mm, yaw_deg = move_to_pose(
        arm, t_robot_target, PLACE_Z_OFFSET, ARM_SPEED_DESCEND_MM_S, piece_name=piece_name
    )
    arm.open_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_AFTER_RELEASE_S)
    arm.set_position(
        x_mm, y_mm, lift_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, yaw_deg,
        speed=ARM_SPEED_TRAVEL_MM_S, is_radian=False, wait=True,
    )


def build_graveyard_pose(
    robot_frame_centers: Dict[int, List[float]],
    t_robot_board: np.ndarray,
    piece_name: str,
) -> np.ndarray:
    """Corner hover pose used when staging captures / exits (anchor square + sideways shift)."""
    graveyard_pose = square_to_robot_pose(
        robot_frame_centers,
        GRAVEYARD_ANCHOR_ROW,
        GRAVEYARD_ANCHOR_COL,
        t_robot_board,
        piece_name=piece_name,
    )
    graveyard_pose[0, 3] += GRAVEYARD_X_SHIFT_M
    return graveyard_pose


def hover_pose(arm: XArmAPI, t_robot_target: np.ndarray) -> None:
    """XY + safe Z only (no plunge)."""
    xyz = t_robot_target[:3, 3]
    x_mm, y_mm, _ = (xyz * 1000.0).tolist()
    safe_z_mm = max(SAFE_Z * 1000.0, MIN_TOOL_Z_M * 1000.0)
    _, _, yaw_deg = Rotation.from_matrix(t_robot_target[:3, :3]).as_euler("xyz", degrees=True)
    yaw_deg = ((yaw_deg + 180.0) % 360.0) - 180.0
    yaw_deg = max(-MAX_ABS_TOOL_YAW_DEG, min(MAX_ABS_TOOL_YAW_DEG, yaw_deg))
    if not np.isfinite([x_mm, y_mm, safe_z_mm, yaw_deg]).all():
        raise RuntimeError(
            f"[pickup] Non-finite pose for hover_pose: "
            f"x={x_mm}, y={y_mm}, z={safe_z_mm}, yaw={yaw_deg}"
        )
    arm.set_position(
        x_mm, y_mm, safe_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, yaw_deg,
        speed=ARM_SPEED_TRAVEL_MM_S, is_radian=False, wait=True,
    )


def build_forward_entry_pose(
    robot_frame_centers: Dict[int, List[float]],
    graveyard_pose: np.ndarray,
) -> np.ndarray:
    """XY nudge from graveyard toward board center. Uses a capped fraction of board depth before and after touches."""
    # Indices 27 and 36 sit near the visual middle of the ranked squares (approximate board centroid XY).
    board_center_xy = np.mean(
        np.array([robot_frame_centers[27][:2], robot_frame_centers[36][:2]], dtype=np.float64),
        axis=0,
    )
    # Corners 3 and 59 span roughly one board diagonal projected to XY for a depth scale.
    board_depth_m = float(
        np.linalg.norm(
            np.array(robot_frame_centers[3][:2], dtype=np.float64)
            - np.array(robot_frame_centers[59][:2], dtype=np.float64)
        )
    )
    if not np.isfinite(board_depth_m) or board_depth_m <= 0.0:
        board_depth_m = 0.4
    step_m = min(board_depth_m * FORWARD_ENTRY_BOARD_FRACTION, MAX_FORWARD_ENTRY_STEP_M)

    # Unit vector from arm base origin toward board_center_xy in the XY plane.
    forward_dir = board_center_xy.copy()
    norm = float(np.linalg.norm(forward_dir))
    if norm < 1e-6:
        forward_dir = np.array([1.0, 0.0], dtype=np.float64)
    else:
        forward_dir /= norm

    entry_pose = graveyard_pose.copy()
    if not np.isfinite(entry_pose[:3, 3]).all():
        return graveyard_pose.copy()
    entry_pose[0, 3] += float(forward_dir[0] * step_m)
    entry_pose[1, 3] += float(forward_dir[1] * step_m)
    if not np.isfinite(entry_pose[:3, 3]).all():
        return graveyard_pose.copy()
    return entry_pose


def move_piece(
    piece_type: str,
    from_square: str,
    to_square: str,
    zed: ZedCamera,
    robot_ip: str = ROBOT_IP_DEFAULT,
) -> None:
    """Capture one frame, run vision, then pick at ``from_square`` and place at ``to_square``."""
    piece_name = normalize_piece_type(piece_type)
    from_row, from_col = algebraic_to_row_col(from_square)
    to_row, to_col = algebraic_to_row_col(to_square)

    arm: Optional[XArmAPI] = None
    graveyard_hover_pose: Optional[np.ndarray] = None
    arm_connected = False
    try:
        img = zed.image
        if img is None:
            raise RuntimeError("No image from ZED.")

        K = zed.camera_intrinsic  # 3x3 pinhole intrinsics from ZED
        vision = build_vision_from_piece_continuity(img, K)
        if vision is None:
            raise RuntimeError(_VISION_TAG_ERR)

        if int(vision.board_state[from_row, from_col]) == 0:
            raise RuntimeError(f"Source square {from_square} appears empty in vision.")

        t_rb = vision.t_robot_board  # board frame in robot base
        from_pose = square_to_robot_pose(
            vision.robot_frame_centers, from_row, from_col, t_rb, piece_name=piece_name
        )
        to_pose = square_to_robot_pose(
            vision.robot_frame_centers, to_row, to_col, t_rb, piece_name=piece_name
        )
        graveyard_hover_pose = build_graveyard_pose(vision.robot_frame_centers, t_rb, piece_name)
        forward_entry_pose = build_forward_entry_pose(
            vision.robot_frame_centers, graveyard_hover_pose
        )

        arm = XArmAPI(robot_ip)
        arm.connect()
        arm_connected = True
        arm.motion_enable(enable=True)
        arm.set_tcp_offset([0, 0, GRIPPER_LENGTH_M * 1000.0, 0, 0, 0])
        arm.set_mode(0)
        arm.set_state(0)
        hover_pose(arm, graveyard_hover_pose)
        hover_pose(arm, forward_entry_pose)
        time.sleep(0.5)
        pickup_pose(arm, from_pose, piece_name=piece_name)
        place_pose(arm, to_pose, piece_name=piece_name)
        # Exit via forward entry, then park away from board.
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


def robot_xyz_pose(
    x_m: float,
    y_m: float,
    z_m: float,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """Build a robot-base pose from fixed XYZ + yaw."""
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix().astype(np.float32)
    pose[:3, 3] = np.array([x_m, y_m, z_m], dtype=np.float32)
    return pose


def replace_promoted_pawn_with_source_queen(
    to_square: str,
    zed: ZedCamera,
    robot_ip: str = ROBOT_IP_DEFAULT,
) -> None:
    """One session: lift pawn off promotion square, stash it, fetch spare queen from fixed source, place queen."""
    if PROMOTION_SOURCE_X_M is None or PROMOTION_SOURCE_Y_M is None or PROMOTION_SOURCE_Z_M is None:
        raise RuntimeError(
            "Set PROMOTION_SOURCE_X_M, PROMOTION_SOURCE_Y_M, and PROMOTION_SOURCE_Z_M in pickup_board_piece.py "
            "before using robot promotions."
        )

    to_row, to_col = algebraic_to_row_col(to_square)
    arm: Optional[XArmAPI] = None
    arm_connected = False
    graveyard_hover_pose: Optional[np.ndarray] = None
    try:
        img = zed.image
        if img is None:
            raise RuntimeError("No image from ZED.")
        K = zed.camera_intrinsic  # 3x3 pinhole intrinsics from ZED
        vision = build_vision_from_piece_continuity(img, K)
        if vision is None:
            raise RuntimeError(_VISION_TAG_ERR)

        t_rb = vision.t_robot_board  # board frame in robot base
        promotion_square_pawn_pose = square_to_robot_pose(
            vision.robot_frame_centers, to_row, to_col, t_rb, piece_name="pawn"
        )
        promotion_square_queen_pose = square_to_robot_pose(
            vision.robot_frame_centers, to_row, to_col, t_rb, piece_name="queen"
        )
        # Board-plane height at promotion square so the discarded pawn lands flush with neighbors.
        source_level_z_m = float(
            square_to_robot_pose(vision.robot_frame_centers, to_row, to_col, t_rb, piece_name=None)[2, 3]
        )
        graveyard_hover_pose = build_graveyard_pose(vision.robot_frame_centers, t_rb, "queen")
        forward_entry_pose = build_forward_entry_pose(vision.robot_frame_centers, graveyard_hover_pose)
        promotion_source_pose = robot_xyz_pose(
            float(PROMOTION_SOURCE_X_M),
            float(PROMOTION_SOURCE_Y_M),
            float(PROMOTION_SOURCE_GRASP_TARGET_Z_M - GRASP_Z_OFFSET),
            yaw_deg=PROMOTION_SOURCE_YAW_DEG,
        )
        promotion_pawn_discard_pose = robot_xyz_pose(
            float(PROMOTION_SOURCE_X_M + PROMOTION_PAWN_DISCARD_X_OFFSET_M),
            float(PROMOTION_SOURCE_Y_M),
            source_level_z_m,
            yaw_deg=PROMOTION_SOURCE_YAW_DEG,
        )

        arm = XArmAPI(robot_ip)
        arm.connect()
        arm_connected = True
        arm.motion_enable(enable=True)
        arm.set_tcp_offset([0, 0, GRIPPER_LENGTH_M * 1000.0, 0, 0, 0])
        arm.set_mode(0)
        arm.set_state(0)

        hover_pose(arm, graveyard_hover_pose)
        hover_pose(arm, forward_entry_pose)
        pickup_pose(arm, promotion_square_pawn_pose, piece_name="pawn")

        hover_pose(arm, promotion_pawn_discard_pose)
        place_pose(arm, promotion_pawn_discard_pose, piece_name="pawn")

        hover_pose(arm, promotion_source_pose)
        pickup_pose(arm, promotion_source_pose, piece_name="queen")

        place_pose(arm, promotion_square_queen_pose, piece_name="queen")

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


def capture_offsets(capture_count, offset_size):
    """3-wide grid shift so multiple victims do not pile on the same spot."""
    graveyard_x = capture_count % 3
    graveyard_y = capture_count // 3
    offset_x = offset_size * graveyard_x
    offset_y = graveyard_y * offset_size
    return offset_x, offset_y

def capture_piece(
    capturing_piece_type: str,
    captured_piece_type: str,
    from_square: str,
    to_square: str,
    zed: ZedCamera,
    capture_count: int = 0,
    robot_ip: str = ROBOT_IP_DEFAULT,
    captured_square: Optional[str] = None,
) -> None:
    """Vision -> remove victim -> graveyard drop -> attacker ``from_square`` -> ``to_square``. En passant: set ``captured_square``."""
    capturing_piece_name = normalize_piece_type(capturing_piece_type)
    captured_piece_name = normalize_piece_type(captured_piece_type)
    from_row, from_col = algebraic_to_row_col(from_square)
    to_row, to_col = algebraic_to_row_col(to_square)
    victim_square = captured_square if captured_square is not None else to_square
    victim_row, victim_col = algebraic_to_row_col(victim_square)

    arm: Optional[XArmAPI] = None
    graveyard_pose: Optional[np.ndarray] = None
    graveyard_hover_pose: Optional[np.ndarray] = None
    arm_connected = False
    try:
        img = zed.image
        if img is None:
            raise RuntimeError("No image from ZED.")

        K = zed.camera_intrinsic  # 3x3 pinhole intrinsics from ZED
        vision = build_vision_from_piece_continuity(img, K)
        if vision is None:
            raise RuntimeError(_VISION_TAG_ERR)

        if int(vision.board_state[from_row, from_col]) == 0:
            raise RuntimeError(f"Source square {from_square} appears empty in vision.")

        t_rb = vision.t_robot_board  # board frame in robot base
        captured_from_pose = square_to_robot_pose(
            vision.robot_frame_centers, victim_row, victim_col, t_rb, piece_name=captured_piece_name
        )
        graveyard_hover_pose = build_graveyard_pose(
            vision.robot_frame_centers, t_rb, captured_piece_name
        )
        forward_entry_pose = build_forward_entry_pose(
            vision.robot_frame_centers, graveyard_hover_pose
        )
        graveyard_pose = square_to_robot_pose(
            vision.robot_frame_centers,
            GRAVEYARD_ANCHOR_ROW,
            GRAVEYARD_ANCHOR_COL,
            t_rb,
            piece_name=captured_piece_name,
        )
        graveyard_pose[0, 3] -= 0.15  # push drop slot slightly toward the arm along board X
        capture_offset_x, capture_offset_y = capture_offsets(capture_count, 0.05)
        graveyard_pose[0, 3] += capture_offset_x
        graveyard_pose[1, 3] += capture_offset_y
        capturing_from_pose = square_to_robot_pose(
            vision.robot_frame_centers, from_row, from_col, t_rb, piece_name=capturing_piece_name
        )
        capturing_to_pose = square_to_robot_pose(
            vision.robot_frame_centers, to_row, to_col, t_rb, piece_name=capturing_piece_name
        )

        arm = XArmAPI(robot_ip)
        arm.connect()
        arm_connected = True
        arm.motion_enable(enable=True)
        arm.set_tcp_offset([0, 0, GRIPPER_LENGTH_M * 1000.0, 0, 0, 0])
        arm.set_mode(0)
        arm.set_state(0)
        hover_pose(arm, graveyard_hover_pose)
        hover_pose(arm, forward_entry_pose)
        time.sleep(0.5)
        pickup_pose(arm, captured_from_pose, piece_name=captured_piece_name)
        hover_pose(arm, forward_entry_pose)
        hover_pose(arm, graveyard_hover_pose)
        place_pose(arm, graveyard_pose, piece_name=captured_piece_name)
        hover_pose(arm, forward_entry_pose)
        pickup_pose(arm, capturing_from_pose, piece_name=capturing_piece_name)
        place_pose(arm, capturing_to_pose, piece_name=capturing_piece_name)
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
