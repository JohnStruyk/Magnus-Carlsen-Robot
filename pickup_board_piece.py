"""
Pick and place chess pieces using ZED + dual AprilTag families (playmat / board).

Square targets are derived from the same warped-board homography as ``detect_pieces``:
warp cell center → raw pixel → camera ray → board plane → robot base. That keeps
arm XY aligned with the rectified grid even when the analytic tag grid differs slightly.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
from piece_continuity import (
    BOARD_CONFIG,
    detect_pieces,
    get_4x4_transform,
    get_board_centers_local,
    get_warped,
)
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera

# ---------------------------------------------------------------------------
# Board / calibration (override in module scope if measured values differ)
# ---------------------------------------------------------------------------
BOARD_TAG_EDGE_M = BOARD_CONFIG["tag_size"]
CHESS_SQUARE_SIZE_M: Optional[float] = None
HAND_EYE_XYZ_BIAS_M = np.zeros(3, dtype=np.float64)

# ---------------------------------------------------------------------------
# Robot motion (Lite6 + parallel gripper)
# ---------------------------------------------------------------------------
ROBOT_IP_DEFAULT = "192.168.1.159"
SAFE_Z = 0.22
GRASP_Z_OFFSET = 0.0001
LIFT_Z_DELTA = 0.06
PLACE_Z_OFFSET = 0.002
TOOL_ROLL_DEG = 180.0
TOOL_PITCH_DEG = 0.0
GRIPPER_LENGTH_M = 0.067
ARM_SPEED_TRAVEL_MM_S = 80
ARM_SPEED_DESCEND_MM_S = 30
GRIPPER_SETTLE_AFTER_OPEN_S = 0.40
GRIPPER_SETTLE_AFTER_CLOSE_S = 0.55
GRIPPER_SETTLE_AFTER_RELEASE_S = 0.40
GRASP_DWELL_BEFORE_CLOSE_S = 0.25

# Chesspiece Calibration
PIECE_CONFIG = {
    "king_height": 0.0950214,
    "queen_height": 0.075184,
    "bishop_height": 0.0638048,
    "knight_height": 0.0569976,
    "rook_height": 0.0455422,
    "pawn_height":  0.0445008
}

# Preview: FROM = yellow, TO = blue (BGR)
_COLOR_FROM_BGR = (0, 255, 255)
_COLOR_TO_BGR = (255, 0, 0)


@dataclass(frozen=True)
class PickVision:
    """Outputs of one camera frame: warped board, occupancy, transforms, homography."""

    warped: np.ndarray
    board_state: np.ndarray
    robot_frame_centers: Dict[int, List[float]]
    t_robot_board: np.ndarray
    playmat_tags: List[Any]
    chessboard_tags: List[Any]
    t_cam_robot: np.ndarray
    split_msg: str
    H_warp_to_img: np.ndarray
    square_px: int


# =============================================================================
# Algebraic notation & board config
# =============================================================================


def algebraic_to_row_col(square: str) -> Tuple[int, int]:
    """Map ``e5`` → ``(row, col)`` for warped image / ``board_state`` (rank 8 → row 0, file a → col 0)."""
    if len(square) != 2 or square[0].lower() not in "abcdefgh" or square[1] not in "12345678":
        raise ValueError(f"Invalid square '{square}'. Use algebraic notation like 'b3'.")
    col = ord(square[0].lower()) - ord("a")
    row = 8 - int(square[1])
    return row, col


def normalize_piece_type(piece_type: str) -> str:
    """Accept PGN-style letters or full names; return canonical piece name."""
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


def _board_config_for_pickup() -> dict:
    """``BOARD_CONFIG`` with optional measured tag edge / square size overrides."""
    cfg = {**BOARD_CONFIG, "tag_size": float(BOARD_TAG_EDGE_M)}
    if CHESS_SQUARE_SIZE_M is not None:
        cfg["square_size"] = float(CHESS_SQUARE_SIZE_M)
    return cfg


# =============================================================================
# Geometry: warped cell → 3D in robot base
# =============================================================================


def warp_cell_center_to_robot_xyz(
    row: int,
    col: int,
    square_px: int,
    H_warp_to_img: np.ndarray,
    K: np.ndarray,
    t_board_to_cam: np.ndarray,
    t_robot_cam: np.ndarray,
) -> Optional[np.ndarray]:
    """
    3D point in robot base for the center of warped cell ``(row, col)``.

    Pipeline: homography to raw pixel → unproject to ray → intersect plane z=0 of the
    board frame (normal from ``t_board_to_cam``) → transform to base with ``t_robot_cam``.

    Returns ``None`` if the ray is parallel to the board (caller uses metric fallback).
    """
    u_w = float(col * square_px + square_px * 0.5)
    v_w = float(row * square_px + square_px * 0.5)
    ph = H_warp_to_img @ np.array([u_w, v_w, 1.0], dtype=np.float64)
    if abs(ph[2]) < 1e-12:
        return None
    u, v = ph[0] / ph[2], ph[1] / ph[2]
    d = np.linalg.inv(K) @ np.array([u, v, 1.0], dtype=np.float64)
    R_bc = t_board_to_cam[:3, :3].astype(np.float64)
    t_bc = t_board_to_cam[:3, 3].astype(np.float64)
    n = R_bc[:, 2]
    denom = float(np.dot(n, d))
    if abs(denom) < 1e-10:
        return None
    alpha = float(np.dot(n, t_bc) / denom)
    p_cam = alpha * d
    return (t_robot_cam @ np.append(p_cam, 1.0))[:3].astype(np.float64)


def compute_robot_frame_centers(
    board_cfg: dict,
    square_px: int,
    H_warp_to_img: np.ndarray,
    K: np.ndarray,
    t_board_to_cam: np.ndarray,
    t_robot_cam: np.ndarray,
) -> Dict[int, List[float]]:
    """
    For each warped cell index ``i = row*8+col``, base-frame XYZ (meters) at square center.

    Primary: ``warp_cell_center_to_robot_xyz``. Fallback: chain through metric grid
    ``get_board_centers_local`` (same as pre-homography fix).
    """
    local = get_board_centers_local(board_cfg)
    out: Dict[int, List[float]] = {}
    idx = 0
    for r in range(8):
        for c in range(8):
            p_fb = (t_robot_cam @ (t_board_to_cam @ local[idx]))[:3]
            p = warp_cell_center_to_robot_xyz(
                r, c, square_px, H_warp_to_img, K, t_board_to_cam, t_robot_cam
            )
            out[idx] = (p if p is not None else p_fb.astype(np.float64)).tolist()
            idx += 1
    return out


def square_to_robot_pose(
    robot_frame_centers: Dict[int, List[float]],
    row: int,
    col: int,
    t_robot_board: np.ndarray,
) -> np.ndarray:
    """4×4 pose in robot base: translation from ``robot_frame_centers``; rotation from board frame."""
    idx = row * 8 + col
    t = np.asarray(robot_frame_centers[idx][:3], dtype=np.float64) + HAND_EYE_XYZ_BIAS_M
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = t_robot_board[:3, :3].astype(np.float32)
    pose[:3, 3] = t.astype(np.float32)
    return pose


# =============================================================================
# Vision: tags → transforms → warped board → centers + occupancy
# =============================================================================


def build_vision_from_piece_continuity(
    img: np.ndarray, camera_intrinsic: np.ndarray
) -> Optional[PickVision]:
    """
    Detect playmat + chessboard tags, solve PnP, warp board, fill ``PickVision``.

    Playmat (``PLAYMAT_TAG_FAMILY``): ``T_cam_robot``. Chessboard (``CHESSBOARD_TAG_FAMILY``):
    ``T_board_to_cam``. Robot point: ``inv(T_cam_robot) @ T_board_to_cam @ p_board``; square
    centers use homography-ray method (see ``compute_robot_frame_centers``).
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

    t_robot_cam = np.linalg.inv(t_cam_robot)
    t_robot_board = t_robot_cam @ t_board_to_cam

    square_px = 100
    warped, _, H_img_to_warp = get_warped(img, b_rvec, b_tvec, camera_intrinsic, square_px=square_px)
    H_warp_to_img = np.linalg.inv(H_img_to_warp)

    centers = compute_robot_frame_centers(
        board_cfg, square_px, H_warp_to_img, camera_intrinsic, t_board_to_cam, t_robot_cam
    )
    board_state = detect_pieces(warped, square_px=square_px)

    return PickVision(
        warped=warped,
        board_state=board_state,
        robot_frame_centers=centers,
        t_robot_board=t_robot_board,
        playmat_tags=list(playmat_raw),
        chessboard_tags=list(chess_raw),
        t_cam_robot=t_cam_robot,
        split_msg=split_msg,
        H_warp_to_img=H_warp_to_img,
        square_px=square_px,
    )


# =============================================================================
# Preview (OpenCV)
# =============================================================================


def _cell_corners_warped(row: int, col: int, square_px: int) -> np.ndarray:
    """One cell as 4×1×2 float32 corners in warped coordinates (TL, TR, BR, BL)."""
    s = float(square_px)
    return np.array(
        [
            [[col * s, row * s], [(col + 1) * s, row * s], [(col + 1) * s, (row + 1) * s], [col * s, (row + 1) * s]]
        ],
        dtype=np.float32,
    )


def _cell_quad_in_raw_image(
    H_warp_to_img: np.ndarray, row: int, col: int, square_px: int
) -> np.ndarray:
    """Project warped cell quad to the original camera image (4×1×2)."""
    return cv2.perspectiveTransform(_cell_corners_warped(row, col, square_px), H_warp_to_img)


def _blend_quad(bgr: np.ndarray, quad_xy: np.ndarray, color_bgr: Tuple[int, int, int], alpha: float = 0.36) -> None:
    """Draw a semi-transparent filled quad + outline on ``bgr`` (in place)."""
    pts = np.round(quad_xy.reshape(4, 2)).astype(np.int32)
    layer = bgr.copy()
    cv2.fillPoly(layer, [pts], color_bgr)
    cv2.addWeighted(layer, alpha, bgr, 1.0 - alpha, 0.0, bgr)
    cv2.polylines(bgr, [pts], True, color_bgr, 3, cv2.LINE_AA)


def show_preview(
    raw_img: np.ndarray,
    warped: np.ndarray,
    camera_intrinsic: np.ndarray,
    from_row: int,
    from_col: int,
    to_row: int,
    to_col: int,
    vision_meta: Optional[Dict[str, Any]] = None,
    from_square: Optional[str] = None,
    to_square: Optional[str] = None,
) -> bool:
    """
    Show warped board (FROM yellow / TO blue) and optional raw view with tag overlays +
    projected quads. Return ``True`` if user pressed ``k`` to confirm execution.
    """
    spx = warped.shape[0] // 8
    vis_warp = warped.copy()

    fx0, fy0 = from_col * spx, from_row * spx
    fx1, fy1 = fx0 + spx, fy0 + spx
    tx0, ty0 = to_col * spx, to_row * spx
    tx1, ty1 = tx0 + spx, ty0 + spx
    cv2.rectangle(vis_warp, (fx0, fy0), (fx1, fy1), _COLOR_FROM_BGR, 3)
    cv2.rectangle(vis_warp, (tx0, ty0), (tx1, ty1), _COLOR_TO_BGR, 3)
    flab = f"FROM {from_square}" if from_square else "FROM"
    tlab = f"TO {to_square}" if to_square else "TO"
    cv2.putText(vis_warp, flab, (fx0 + 4, fy0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_FROM_BGR, 2)
    cv2.putText(vis_warp, tlab, (tx0 + 4, ty0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_TO_BGR, 2)

    max_dim = 900
    h, w = vis_warp.shape[:2]
    scale = min(max_dim / float(max(h, w)), 1.0)
    if scale < 1.0:
        vis_warp = cv2.resize(vis_warp, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    if vision_meta and vision_meta.get("playmat_tags") is not None:
        tag_vis = to_bgr_display(raw_img)
        if tag_vis is not None:
            tag_vis = tag_vis.copy()
            draw_dual_family_tag_overlays(
                tag_vis, vision_meta["playmat_tags"], vision_meta["chessboard_tags"]
            )
            tcr = vision_meta.get("t_cam_robot")
            if tcr is not None:
                draw_pose_axes(tag_vis, camera_intrinsic, tcr, size=TAG_SIZE)
            H_wi = vision_meta.get("H_warp_to_img")
            spx_meta = int(vision_meta.get("square_px", spx))
            if H_wi is not None:
                _blend_quad(tag_vis, _cell_quad_in_raw_image(H_wi, to_row, to_col, spx_meta), _COLOR_TO_BGR, 0.32)
                _blend_quad(tag_vis, _cell_quad_in_raw_image(H_wi, from_row, from_col, spx_meta), _COLOR_FROM_BGR, 0.36)
                cf = _cell_quad_in_raw_image(H_wi, from_row, from_col, spx_meta).reshape(4, 2).mean(axis=0)
                ct = _cell_quad_in_raw_image(H_wi, to_row, to_col, spx_meta).reshape(4, 2).mean(axis=0)
                cv2.putText(
                    tag_vis, flab, (max(4, int(cf[0]) - 40), max(22, int(cf[1]))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, _COLOR_FROM_BGR, 2, cv2.LINE_AA,
                )
                cv2.putText(
                    tag_vis, tlab, (max(4, int(ct[0]) - 40), max(22, int(ct[1]))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, _COLOR_TO_BGR, 2, cv2.LINE_AA,
                )
            line = f"pickup: {vision_meta.get('split_msg', '')} | yellow=FROM blue=TO (raw = arm targets)"
            for dy, color, th in ((2, (255, 255, 255), 2), (0, (0, 200, 0), 1)):
                cv2.putText(tag_vis, line, (12, 86 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, th, cv2.LINE_AA)
            cv2.namedWindow("pickup: raw camera (FROM yellow / TO blue)", cv2.WINDOW_NORMAL)
            cv2.imshow("pickup: raw camera (FROM yellow / TO blue)", resize_for_preview(tag_vis))

    cv2.namedWindow("Warped board", cv2.WINDOW_NORMAL)
    cv2.imshow("Warped board", vis_warp)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key == ord("k")


# =============================================================================
# Arm trajectories
# =============================================================================


def move_to_pose(
    arm: XArmAPI, t_robot_target: np.ndarray, z_offset_m: float, descend_speed: int
) -> Tuple[float, float, float, float]:
    """Travel at ``SAFE_Z``, descend to target Z + offset, return mm coords and yaw for lift."""
    xyz = t_robot_target[:3, 3]
    x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
    safe_z_mm = SAFE_Z * 1000.0
    target_z_mm = z_mm + z_offset_m * 1000.0
    lift_z_mm = max(safe_z_mm, target_z_mm + LIFT_Z_DELTA * 1000.0)
    _, _, yaw_deg = Rotation.from_matrix(t_robot_target[:3, :3]).as_euler("xyz", degrees=True)

    arm.set_position(
        x_mm, y_mm, safe_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, yaw_deg,
        speed=ARM_SPEED_TRAVEL_MM_S, is_radian=False, wait=True,
    )
    arm.set_position(
        x_mm, y_mm, target_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, yaw_deg,
        speed=descend_speed, is_radian=False, wait=True,
    )
    return x_mm, y_mm, lift_z_mm, yaw_deg


def pickup_pose(arm: XArmAPI, t_robot_target: np.ndarray) -> None:
    """Open gripper, descend to grasp height, close, lift slightly."""
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
        speed=ARM_SPEED_TRAVEL_MM_S, is_radian=False, wait=True,
    )


def place_pose(arm: XArmAPI, t_robot_target: np.ndarray) -> None:
    """Descend to place height, open gripper, retract to safe Z."""
    x_mm, y_mm, lift_z_mm, yaw_deg = move_to_pose(
        arm, t_robot_target, PLACE_Z_OFFSET, ARM_SPEED_DESCEND_MM_S
    )
    arm.open_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_AFTER_RELEASE_S)
    arm.set_position(
        x_mm, y_mm, lift_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, yaw_deg,
        speed=ARM_SPEED_TRAVEL_MM_S, is_radian=False, wait=True,
    )


# =============================================================================
# End-to-end pick / place
# =============================================================================


def move_piece(
    piece_type: str,
    from_square: str,
    to_square: str,
    robot_ip: str = ROBOT_IP_DEFAULT,
    preview: bool = False,
) -> None:
    """Capture one frame, run vision, optionally preview, then pick at ``from_square`` and place at ``to_square``."""
    piece_name = normalize_piece_type(piece_type)
    from_row, from_col = algebraic_to_row_col(from_square)
    to_row, to_col = algebraic_to_row_col(to_square)

    zed = ZedCamera()
    arm: Optional[XArmAPI] = None
    try:
        print("[pickup] Capturing camera image...")
        img = zed.image
        if img is None:
            raise RuntimeError("No image from ZED.")

        print("[pickup] Running vision (playmat + chessboard tags)...")
        K = zed.camera_intrinsic
        vision = build_vision_from_piece_continuity(img, K)
        if vision is None:
            raise RuntimeError(
                f"Vision failed: need four tags ids 0–3 on {PLAYMAT_TAG_FAMILY} (playmat) "
                f"and four on {CHESSBOARD_TAG_FAMILY} (chessboard). See console above."
            )

        execute = True
        if preview:
            print("[pickup] Preview: press 'k' to execute, any other key to cancel.")
            execute = show_preview(
                img,
                vision.warped,
                K,
                from_row,
                from_col,
                to_row,
                to_col,
                vision_meta={
                    "playmat_tags": vision.playmat_tags,
                    "chessboard_tags": vision.chessboard_tags,
                    "t_cam_robot": vision.t_cam_robot,
                    "split_msg": vision.split_msg,
                    "H_warp_to_img": vision.H_warp_to_img,
                    "square_px": vision.square_px,
                },
                from_square=from_square,
                to_square=to_square,
            )

        if int(vision.board_state[from_row, from_col]) == 0:
            raise RuntimeError(f"Source square {from_square} appears empty in vision.")

        t_rb = vision.t_robot_board
        from_pose = square_to_robot_pose(vision.robot_frame_centers, from_row, from_col, t_rb)
        to_pose = square_to_robot_pose(vision.robot_frame_centers, to_row, to_col, t_rb)

        print(f"Moving {piece_name} {from_square} → {to_square}")
        print(
            f"[pickup] Indices row*8+col: FROM ({from_row},{from_col})={from_row * 8 + from_col}, "
            f"TO ({to_row},{to_col})={to_row * 8 + to_col}"
        )
        print(f"From xyz (m): {from_pose[:3, 3].tolist()}")
        print(f"To xyz (m): {to_pose[:3, 3].tolist()}")

        if not execute:
            print("Cancelled (preview: press 'k' to run).")
            return

        print("[pickup] Connecting to arm...")
        arm = XArmAPI(robot_ip)
        arm.connect()
        arm.motion_enable(enable=True)
        arm.set_tcp_offset([0, 0, GRIPPER_LENGTH_M * 1000.0, 0, 0, 0])
        arm.set_mode(0)
        arm.set_state(0)
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        print("[pickup] Executing pick / place...")
        pickup_pose(arm, from_pose)
        place_pose(arm, to_pose)
    finally:
        if arm is not None:
            arm.stop_lite6_gripper()
            arm.move_gohome(wait=True)
            time.sleep(0.5)
            arm.disconnect()
        zed.close()
        cv2.destroyAllWindows()


def move_piece_three_params(piece_type: str, from_square: str, to_square: str) -> None:
    """Thin wrapper for callers that only pass ``(piece_type, from_square, to_square)``."""
    move_piece(piece_type, from_square, to_square)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pick and place a chess piece (ZED + AprilTags + Lite6).")
    p.add_argument("--piece-type", required=True, help="pawn/knight/bishop/rook/queen/king or P/N/B/R/Q/K")
    p.add_argument("--from-square", required=True, help="Source square, e.g. e5")
    p.add_argument("--to-square", required=True, help="Destination square, e.g. e4")
    p.add_argument("--robot-ip", default=ROBOT_IP_DEFAULT)
    p.add_argument("--preview", action="store_true", help="Show overlays; press 'k' to execute.")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    move_piece(a.piece_type, a.from_square, a.to_square, robot_ip=a.robot_ip, preview=a.preview)


if __name__ == "__main__":
    main()
