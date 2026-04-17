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


BOARD_TAG_EDGE_M = BOARD_CONFIG["tag_size"]
# Optional: set to ruler-measured square size if it differs from piece_continuity BOARD_CONFIG.
CHESS_SQUARE_SIZE_M = None

# Last-mile bias in robot/playmat frame (meters): tune if XY is consistently offset or Z hits the table.
HAND_EYE_XYZ_BIAS_M = np.array([0.0, 0.0, 0.0], dtype=np.float64)

ROBOT_IP_DEFAULT = "192.168.1.159"
# Minimum height (m) for horizontal moves; increase if the tool still touches the table during travel.
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

# Preview: BGR — pick (FROM) = yellow, place (TO) = blue (matches user request).
COLOR_FROM_BGR = (0, 255, 255)
COLOR_TO_BGR = (255, 0, 0)


def _square_quad_warped(row, col, square_px):
    """Corners of one cell in warped pixel coords: TL, TR, BR, BL (x, y)."""
    s = float(square_px)
    return np.array(
        [
            [
                [col * s, row * s],
                [(col + 1) * s, row * s],
                [(col + 1) * s, (row + 1) * s],
                [col * s, (row + 1) * s],
            ]
        ],
        dtype=np.float32,
    )


def _quad_warped_to_raw(H_warp_to_img, row, col, square_px):
    """Map a warped-board cell to a quadrilateral in the original camera image."""
    qw = _square_quad_warped(row, col, square_px)
    return cv2.perspectiveTransform(qw, H_warp_to_img)


def _blend_quad_on_image(bgr, quad_xy, color_bgr, alpha=0.38, edge_thickness=3):
    """Semi-transparent fill + solid outline on ``bgr`` (modified in place)."""
    pts = np.round(quad_xy.reshape(4, 2)).astype(np.int32)
    layer = bgr.copy()
    cv2.fillPoly(layer, [pts], color_bgr)
    cv2.addWeighted(layer, alpha, bgr, 1.0 - alpha, 0.0, bgr)
    cv2.polylines(bgr, [pts], True, color_bgr, edge_thickness, cv2.LINE_AA)


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


def _board_config_for_pickup():
    """BOARD_CONFIG with measured chessboard tag edge (and optional square size)."""
    cfg = {**BOARD_CONFIG, "tag_size": float(BOARD_TAG_EDGE_M)}
    if CHESS_SQUARE_SIZE_M is not None:
        cfg["square_size"] = float(CHESS_SQUARE_SIZE_M)
    return cfg


def _warp_cell_center_to_robot_xyz(
    row: int,
    col: int,
    square_px: int,
    H_warp_to_img: np.ndarray,
    K: np.ndarray,
    t_board_to_cam: np.ndarray,
    t_robot_cam: np.ndarray,
):
    """
    Map the same warped-cell center that ``detect_pieces`` uses to a 3D point in the robot
    base frame by: warp pixel → raw pixel → ray through camera → intersect board plane
    (from ``t_board_to_cam``) → ``t_robot_cam``.

    This avoids a mismatch between the 4-corner homography warp and a separate uniform
    metric grid from ``get_board_centers_local``, which can look fine in preview but skew XY.

    Returns None if the ray is degenerate w.r.t. the board plane (caller should fallback).
    """
    u_w = float(col * square_px + square_px * 0.5)
    v_w = float(row * square_px + square_px * 0.5)
    pw = np.array([u_w, v_w, 1.0], dtype=np.float64)
    ph = H_warp_to_img @ pw
    if abs(ph[2]) < 1e-12:
        return None
    u = ph[0] / ph[2]
    v = ph[1] / ph[2]
    d = np.linalg.inv(K) @ np.array([u, v, 1.0], dtype=np.float64)
    R_bc = t_board_to_cam[:3, :3].astype(np.float64)
    t_bc = t_board_to_cam[:3, 3].astype(np.float64)
    n = R_bc[:, 2]
    denom = float(np.dot(n, d))
    if abs(denom) < 1e-10:
        return None
    alpha = float(np.dot(n, t_bc) / denom)
    p_cam = alpha * d
    p_cam_h = np.append(p_cam, 1.0)
    p_robot = (t_robot_cam @ p_cam_h)[:3]
    return np.asarray(p_robot, dtype=np.float64)


def square_to_robot_pose(robot_frame_centers, row, col, t_robot_board):
    """
    Index ``row*8+col`` matches ``detect_pieces`` / warped board (rank 8 at top row, file a
    at col 0). Centers in ``robot_frame_centers`` are homography-ray hits on the board plane.
    """
    idx = row * 8 + col
    p_robot = np.asarray(robot_frame_centers[idx][:3], dtype=np.float64) + HAND_EYE_XYZ_BIAS_M
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = t_robot_board[:3, :3].astype(np.float32)
    pose[:3, 3] = p_robot.astype(np.float32)
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
    board_cfg = _board_config_for_pickup()

    t_cam_robot = get_transform_camera_robot_from_tags(playmat_tags, camera_intrinsic)
    if t_cam_robot is None:
        return None

    t_board_to_cam, b_rvec, b_tvec = get_4x4_transform(
        board_tags, board_cfg, camera_intrinsic, strict=True
    )
    if t_board_to_cam is None:
        print("[pickup] Board PnP failed (need 4 board corners visible).")
        return None

    # checkpoint1: t_robot_cam = inv(camera_pose) with camera_pose = T_cam_robot
    t_robot_cam = np.linalg.inv(t_cam_robot)
    t_robot_board = t_robot_cam @ t_board_to_cam

    square_px = 100
    warped, _img_corners, H_img_to_warp = get_warped(
        img, b_rvec, b_tvec, camera_intrinsic, square_px=square_px
    )
    # findHomography(img_pts, warp_pts): p_warp ~ H @ p_img → p_img = H^{-1} @ p_warp
    H_warp_to_img = np.linalg.inv(H_img_to_warp)

    # Robot XY follows the same warped grid as detect_pieces / preview quads (ray through
    # raw pixel, intersect board plane). Pure metric-grid centers can disagree with that warp.
    local_centers = get_board_centers_local(board_cfg)
    robot_frame_centers = {}
    idx = 0
    for r in range(8):
        for c in range(8):
            p_fb = (t_robot_cam @ (t_board_to_cam @ local_centers[idx]))[:3]
            p_robot = _warp_cell_center_to_robot_xyz(
                r,
                c,
                square_px,
                H_warp_to_img,
                camera_intrinsic,
                t_board_to_cam,
                t_robot_cam,
            )
            if p_robot is None:
                p_robot = p_fb.astype(np.float64)
            robot_frame_centers[idx] = p_robot.tolist()
            idx += 1
    board_state = detect_pieces(warped, square_px=square_px)
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
        "H_warp_to_img": H_warp_to_img,
        "square_px": square_px,
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
    from_square=None,
    to_square=None,
):
    """
    Warped board: yellow = FROM (pick), blue = TO (place).
    Raw camera: same cells projected with the board homography (inverse of warp) so you see
    exactly which physical quadrilaterals match the arm targets. Press 'k' to execute.
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
    cv2.rectangle(preview_img, (fx0, fy0), (fx1, fy1), COLOR_FROM_BGR, 3)
    cv2.rectangle(preview_img, (tx0, ty0), (tx1, ty1), COLOR_TO_BGR, 3)
    flab = f"FROM {from_square}" if from_square else "FROM"
    tlab = f"TO {to_square}" if to_square else "TO"
    cv2.putText(preview_img, flab, (fx0 + 4, fy0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_FROM_BGR, 2)
    cv2.putText(preview_img, tlab, (tx0 + 4, ty0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TO_BGR, 2)

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
            H_warp_to_img = vision_meta.get("H_warp_to_img")
            spx = vision_meta.get("square_px", square_px)
            if H_warp_to_img is not None:
                # Destination (blue) under pick (yellow) so both stay visible.
                q_to = _quad_warped_to_raw(H_warp_to_img, to_row, to_col, spx)
                q_from = _quad_warped_to_raw(H_warp_to_img, from_row, from_col, spx)
                _blend_quad_on_image(tag_vis, q_to, COLOR_TO_BGR, alpha=0.32)
                _blend_quad_on_image(tag_vis, q_from, COLOR_FROM_BGR, alpha=0.36)
                c_from = q_from.reshape(4, 2).mean(axis=0)
                c_to = q_to.reshape(4, 2).mean(axis=0)
                cf = (int(c_from[0]), int(c_from[1]))
                ct = (int(c_to[0]), int(c_to[1]))
                cv2.putText(
                    tag_vis,
                    flab,
                    (max(4, cf[0] - 40), max(22, cf[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    COLOR_FROM_BGR,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    tag_vis,
                    tlab,
                    (max(4, ct[0] - 40), max(22, ct[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    COLOR_TO_BGR,
                    2,
                    cv2.LINE_AA,
                )
            status = vision_meta.get("split_msg", "")
            line = f"pickup: {status} | yellow=FROM blue=TO (raw = arm targets)"
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
            cv2.namedWindow("pickup: raw camera (FROM yellow / TO blue)", cv2.WINDOW_NORMAL)
            cv2.imshow("pickup: raw camera (FROM yellow / TO blue)", resize_for_preview(tag_vis))

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
                    "H_warp_to_img": vision["H_warp_to_img"],
                    "square_px": vision["square_px"],
                },
                from_square=from_square,
                to_square=to_square,
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
        print(
            f"[pickup] Grid index row*8+col: FROM=({from_row},{from_col})→{from_row * 8 + from_col}, "
            f"TO=({to_row},{to_col})→{to_row * 8 + to_col} (must match warped yellow/blue cells)"
        )
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
