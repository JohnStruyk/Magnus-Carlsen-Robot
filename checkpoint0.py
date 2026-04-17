import cv2, numpy
from pupil_apriltags import Detector

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera

TAG_SIZE = 0.08
PREVIEW_MAX_WIDTH = 1280

# Different families, same numeric ids 0–3: decode separately so both can appear in one image.
# tag36h11 ≈ 6×6; tag25h9 ≈ 5×5. Swap if your physical prints are reversed.
PLAYMAT_TAG_FAMILY = "tag36h11"
CHESSBOARD_TAG_FAMILY = "tag25h9"
APRILTAG_FAMILY = PLAYMAT_TAG_FAMILY

PLAYMAT_TAG_IDS = (0, 1, 2, 3)

# Playmat corners in robot frame — ids 0–3 in PLAYMAT_TAG_FAMILY
TAG_CENTER_COORDINATES = [[0.38, 0.4],
                         [0.38, -0.4],
                         [0.0, 0.4],
                         [0.0, -0.4]]

def get_pnp_pairs(tags):
    """3D–2D pairs for playmat tags (PLAYMAT_TAG_FAMILY), ids 0–3.

    World corner order **must match** ``piece_continuity.get_4x4_transform`` (same
    ``tag.corners[k]`` → board/playmat XY mapping). The old order swapped Y for
    corners 0–3 vs that function, which made camera↔robot PnP inconsistent with
    chessboard PnP: overlays looked correct but ``t_robot_cam @ t_board_to_cam``
    sent the arm to the wrong XY.
    """
    half = TAG_SIZE / 2.0
    world_points = numpy.empty([0, 3])
    image_points = numpy.empty([0, 2])

    for tag in tags:
        tid = int(tag.tag_id)
        if tid < 0 or tid > 3:
            continue
        cx, cy = TAG_CENTER_COORDINATES[tid]
        # Identical to piece_continuity.get_4x4_transform wp_corners (indices 0..3).
        wp_corners = [
            [cx - half, cy - half],
            [cx - half, cy + half],
            [cx + half, cy + half],
            [cx + half, cy - half],
        ]
        for k in range(4):
            wp = numpy.array([wp_corners[k][0], wp_corners[k][1], 0.0], dtype=numpy.float64)
            ip = numpy.asarray(tag.corners[k], dtype=numpy.float64)
            world_points = numpy.vstack([world_points, wp])
            image_points = numpy.vstack([image_points, ip])

    return world_points, image_points


def _tag_polygon_area_sq_px(tag):
    """Quadrilateral area in pixels (opencv expects Nx1x2)."""
    c = numpy.asarray(tag.corners, dtype=numpy.float32)
    if c.shape[0] < 4:
        return 0.0
    c = c.reshape(-1, 1, 2)
    return float(abs(cv2.contourArea(c)))


def to_bgr_display(image):
    """BGR image for cv2.imshow (handles BGRA / BGR / gray from ZED)."""
    if image is None:
        return None
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _to_gray(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_apriltags_gray(image, families=None):
    """Return (gray_uint8, tags). Safe grayscale conversion."""
    if image is None:
        return None, []
    fam = families or APRILTAG_FAMILY
    gray = _to_gray(image)
    detector = Detector(families=fam)
    tags = detector.detect(gray, estimate_tag_pose=False)
    return gray, tags


def detect_playmat_and_chessboard_tags(image):
    """
    Run two detectors: playmat (e.g. tag36h11) and chessboard (e.g. tag25h9). Both use ids 0–3.

    Returns (gray, playmat_tags, chessboard_tags).
    """
    if image is None:
        return None, [], []
    gray = _to_gray(image)
    playmat_tags = Detector(families=PLAYMAT_TAG_FAMILY).detect(gray, estimate_tag_pose=False)
    chess_tags = Detector(families=CHESSBOARD_TAG_FAMILY).detect(gray, estimate_tag_pose=False)
    return gray, playmat_tags, chess_tags


def best_tag_per_id_0_3(tags):
    """
    One detection per id 0–3 (largest polygon if duplicates). Returns (list, ok) with ok True iff len==4.
    """
    by_id = {}
    for t in tags:
        tid = int(t.tag_id)
        if tid not in (0, 1, 2, 3):
            continue
        prev = by_id.get(tid)
        if prev is None or _tag_polygon_area_sq_px(t) > _tag_polygon_area_sq_px(prev):
            by_id[tid] = t
    ordered = [by_id[i] for i in (0, 1, 2, 3) if i in by_id]
    return ordered, len(by_id) == 4


def draw_all_tag_overlays(bgr_image, tags):
    """
    Draw every detected tag: outline, center, ID (and hamming if present).
    Mutates bgr_image in place.
    """
    for tag in tags:
        tid = int(tag.tag_id)
        corners = numpy.asarray(tag.corners, dtype=numpy.int32)
        if corners.shape[0] >= 4:
            for i in range(4):
                p0 = tuple(corners[i])
                p1 = tuple(corners[(i + 1) % 4])
                cv2.line(bgr_image, p0, p1, (0, 255, 0), 2)
        cx, cy = int(tag.center[0]), int(tag.center[1])
        label = f"id:{tid}"
        if getattr(tag, "hamming", None) is not None:
            label += f" h:{tag.hamming}"
        if getattr(tag, "decision_margin", None) is not None:
            label += f" dm:{float(tag.decision_margin):.0f}"
        cv2.circle(bgr_image, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(
            bgr_image,
            label,
            (cx + 8, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    n_ids = sum(1 for t in tags if 0 <= int(t.tag_id) <= 3)
    summary = f"tags={len(tags)} (ids 0-3 count={n_ids}) family={APRILTAG_FAMILY}"
    cv2.putText(
        bgr_image,
        summary,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        bgr_image,
        summary,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def draw_dual_family_tag_overlays(bgr_image, playmat_tags, chessboard_tags):
    """Draw playmat (green) vs chessboard (orange) detections; same ids 0–3, different families."""
    def draw_set(tags, prefix, color):
        for tag in tags:
            tid = int(tag.tag_id)
            corners = numpy.asarray(tag.corners, dtype=numpy.int32)
            if corners.shape[0] >= 4:
                for i in range(4):
                    p0 = tuple(corners[i])
                    p1 = tuple(corners[(i + 1) % 4])
                    cv2.line(bgr_image, p0, p1, color, 2)
            cx, cy = int(tag.center[0]), int(tag.center[1])
            label = f"{prefix}{tid}"
            cv2.circle(bgr_image, (cx, cy), 5, color, -1)
            cv2.putText(
                bgr_image,
                label,
                (cx + 6, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    draw_set(playmat_tags, "pm", (0, 220, 0))
    draw_set(chessboard_tags, "ch", (0, 140, 255))
    summary = (
        f"pm={PLAYMAT_TAG_FAMILY} n={len(playmat_tags)} | "
        f"ch={CHESSBOARD_TAG_FAMILY} n={len(chessboard_tags)}"
    )
    cv2.putText(bgr_image, summary, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bgr_image, summary, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)


def resize_for_preview(bgr, max_w=PREVIEW_MAX_WIDTH):
    h, w = bgr.shape[:2]
    if w <= max_w:
        return bgr
    scale = max_w / float(w)
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def get_transform_camera_robot_from_tags(tags, camera_intrinsic):
    """PnP from playmat tags (ids 0–3); pass one tag per id from best_tag_per_id_0_3."""
    world_points, image_points = get_pnp_pairs(tags)
    if world_points.shape[0] < 4:
        print("Insufficient playmat tag corners after filtering (need family %s, ids 0-3)." % PLAYMAT_TAG_FAMILY)
        return None
    success, rotation_vec, translation = cv2.solvePnP(
        world_points, image_points, camera_intrinsic, None
    )
    if success is not True:
        print("PnP Calculation Failed.")
        return None
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    transform_mat = numpy.eye(4)
    transform_mat[:3, :3] = rotation_mat
    transform_mat[:3, 3] = translation.flatten()
    return transform_mat


def get_transform_camera_robot(observation, camera_intrinsic, tags=None):
    """Camera <- robot (playmat) from AprilTag PnP. Optional ``tags`` skips re-detection."""

    if tags is None:
        _, playmat_raw, _ = detect_playmat_and_chessboard_tags(observation)
        tags, ok = best_tag_per_id_0_3(playmat_raw)
        if not ok:
            print(
                f"Playmat family {PLAYMAT_TAG_FAMILY}: need four tags with ids 0-3; "
                f"got ids {sorted(set(int(t.tag_id) for t in playmat_raw))} (n={len(playmat_raw)})"
            )
            return None
    print(f"Playmat tags for PnP: {len(tags)} (family {PLAYMAT_TAG_FAMILY})")
    if tags:
        ids = sorted(set(int(t.tag_id) for t in tags))
        print(f"Playmat tag ids: {ids}")
    world_points, image_points = get_pnp_pairs(tags)
    if world_points.shape[0] < 4:
        print(f'Insufficient valid tag corners found.')
        return None

    # Get Transformation
    success, rotation_vec, translation = cv2.solvePnP(world_points, image_points, camera_intrinsic, None)
    if success is not True:
        print('PnP Calculation Failed.')
        return None
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    transform_mat = numpy.eye(4)
    transform_mat[:3, :3] = rotation_mat
    transform_mat[:3, 3] = translation.flatten()

    return transform_mat

def main():
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    try:
        cv_image = zed.image
        if cv_image is None:
            print("No image from camera.")
            return

        _, playmat_tags, chess_tags = detect_playmat_and_chessboard_tags(cv_image)
        vis = to_bgr_display(cv_image)
        draw_dual_family_tag_overlays(vis, playmat_tags, chess_tags)

        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic, tags=None)
        if t_cam_robot is not None:
            draw_pose_axes(vis, camera_intrinsic, t_cam_robot, size=TAG_SIZE)
            status = "PnP OK — pose axes (playmat %s)" % PLAYMAT_TAG_FAMILY
            color = (0, 255, 0)
        else:
            status = "PnP FAILED — need four playmat tags ids 0-3 (%s)" % PLAYMAT_TAG_FAMILY
            color = (0, 0, 255)

        cv2.putText(
            vis,
            status,
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

        print(
            f"[checkpoint0] Preview: playmat n={len(playmat_tags)} chess n={len(chess_tags)}. {status}"
        )
        print("[checkpoint0] Press any key to close.")

        cv2.namedWindow("checkpoint0: AprilTag preview", cv2.WINDOW_NORMAL)
        cv2.imshow("checkpoint0: AprilTag preview", resize_for_preview(vis))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        zed.close()


if __name__ == "__main__":
    main()
