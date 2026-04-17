import cv2, numpy
from pupil_apriltags import Detector

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera

TAG_SIZE = 0.08
APRILTAG_FAMILY = "tag36h11"
PREVIEW_MAX_WIDTH = 1280

# Playmat / robot calibration uses AprilTag ids 4–7 (see get_pnp_pairs).
PLAYMAT_TAG_IDS = (4, 5, 6, 7)

# top-left, top-right, bottom-left, bottom-right — same layout as before; index i pairs with tag id 4+i
TAG_CENTER_COORDINATES = [[0.38, 0.4],
                         [0.38, -0.4],
                         [0.0, 0.4],
                         [0.0, -0.4]]

def get_pnp_pairs(tags):
    """
    Extract corresponding 3D world coordinates and 2D image coordinates for 
    the corners of detected AprilTags on the **playmat** (ids 4–7).

    Tag id 4 uses TAG_CENTER_COORDINATES[0], id 5 uses [1], etc.

    Parameters
    ----------
    tags : list
        A list of AprilTag detection objects returned by the pupil_apriltags detector.

    Returns
    -------
    world_points : numpy.ndarray
        An (N, 3) array of 3D world coordinates for the tag corners.
    image_points : numpy.ndarray
        An (N, 2) array of corresponding 2D image pixel coordinates for the tag corners.
    """
    world_points = numpy.empty([0, 3])
    image_points = numpy.empty([0, 2])

    for tag in tags:
        tid = int(tag.tag_id)
        if tid < 4 or tid > 7:
            continue
        tag_center = TAG_CENTER_COORDINATES[tid - 4]

        # Bottom left corner
        wp = numpy.zeros(3)
        wp[0] = tag_center[0] - (TAG_SIZE / 2)
        wp[1] = tag_center[1] + (TAG_SIZE / 2)

        ip = tag.corners[0]

        world_points = numpy.vstack([world_points, wp])
        image_points = numpy.vstack([image_points, ip])

        # Bottom right corner
        wp = numpy.zeros(3)
        wp[0] = tag_center[0] - (TAG_SIZE / 2)
        wp[1] = tag_center[1] - (TAG_SIZE / 2)

        ip = tag.corners[1]

        world_points = numpy.vstack([world_points, wp])
        image_points = numpy.vstack([image_points, ip])

        # Top right corner
        wp = numpy.zeros(3)
        wp[0] = tag_center[0] + (TAG_SIZE / 2)
        wp[1] = tag_center[1] - (TAG_SIZE / 2)

        ip = tag.corners[2]

        world_points = numpy.vstack([world_points, wp])
        image_points = numpy.vstack([image_points, ip])

        # Top left corner
        wp = numpy.zeros(3)
        wp[0] = tag_center[0] + (TAG_SIZE / 2)
        wp[1] = tag_center[1] + (TAG_SIZE / 2)

        ip = tag.corners[3]

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


def detect_apriltags_gray(image, families=None):
    """Return (gray_uint8, tags). Safe grayscale conversion."""
    if image is None:
        return None, []
    fam = families or APRILTAG_FAMILY
    if len(image.shape) == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = Detector(families=fam)
    tags = detector.detect(gray, estimate_tag_pose=False)
    return gray, tags


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
    n_playmat = sum(1 for t in tags if 4 <= int(t.tag_id) <= 7)
    summary = f"tags={len(tags)} (playmat ids 4-7 count={n_playmat}) family={APRILTAG_FAMILY}"
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


def resize_for_preview(bgr, max_w=PREVIEW_MAX_WIDTH):
    h, w = bgr.shape[:2]
    if w <= max_w:
        return bgr
    scale = max_w / float(w)
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def partition_playmat_and_board_tags(
    tags,
    chessboard_corner_tag_ids=(0, 1, 2, 3),
    playmat_tag_ids=(4, 5, 6, 7),
):
    """
    Split detections into playmat tags (default ids 4–7, robot calibration) and chessboard
    corner tags (default ids 0–3, piece_continuity BOARD_CONFIG).

    - Case 1: All four chessboard ids and all four playmat ids appear at least once:
      one detection per id (largest area wins if duplicated).
    - Case 2: No playmat ids in the image — duplicate chessboard_corner_tag_ids on mat+board
      (two prints per id); larger polygon → playmat, smaller → board.

    ``board_tag_ids`` is deprecated; use ``chessboard_corner_tag_ids`` + ``playmat_tag_ids``.

    Returns (playmat_tags, board_tags, debug_msg). On failure, tags are None and msg explains.
    """
    by_id = {}
    for t in tags:
        tid = int(t.tag_id)
        by_id.setdefault(tid, []).append(t)

    cids = tuple(chessboard_corner_tag_ids)
    pids = tuple(playmat_tag_ids)

    # Case 1: distinct playmat (4–7) and chessboard (0–3)
    if all(cid in by_id for cid in cids) and all(pid in by_id for pid in pids):
        playmat = []
        board = []
        for pid in pids:
            cands = sorted(by_id[pid], key=_tag_polygon_area_sq_px, reverse=True)
            playmat.append(cands[0])
        for cid in cids:
            cands = sorted(by_id[cid], key=_tag_polygon_area_sq_px, reverse=True)
            board.append(cands[0])
        return playmat, board, "split: playmat 4-7, chessboard 0-3"

    # Case 2: duplicate chessboard ids only (same ids on mat + board; no 4–7 playmat tags)
    if not any(pid in by_id for pid in pids):
        playmat = []
        board = []
        for i in sorted(cids):
            if i not in by_id:
                return None, None, f"missing chessboard tag id {i} (duplicate mode)"
            cands = sorted(by_id[i], key=_tag_polygon_area_sq_px, reverse=True)
            playmat.append(cands[0])
            if len(cands) < 2:
                return (
                    None,
                    None,
                    f"tag id {i}: only one detection; use playmat tags {list(pids)} on the mat "
                    f"or print duplicate {list(cids)} on board+mat",
                )
            board.append(cands[1])
        return playmat, board, "split: duplicate chessboard ids — larger area → playmat, smaller → board"

    return (
        None,
        None,
        f"need all of playmat {list(pids)} and chessboard {list(cids)}, or duplicate-only chessboard ids",
    )


def get_transform_camera_robot_from_tags(tags, camera_intrinsic):
    """
    Same as get_transform_camera_robot but uses an existing tag list (no second detect).
    Pass the playmat tags (ids 4–7, one per id) used for robot-frame PnP.
    """
    world_points, image_points = get_pnp_pairs(tags)
    if world_points.shape[0] < 4:
        print("Insufficient playmat tag corners after filtering.")
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
    """
    Calculate the 4x4 transformation matrix from the camera frame to the 
    robot base frame using AprilTag detections.

    The function detects AprilTags in the provided image, retrieves 
    the 3D-2D point correspondences, and uses the Perspective-n-Point (PnP) algorithm 
    to estimate the pose of the camera.

    Parameters
    ----------
    observation : numpy.ndarray
        The input image from the camera. Can be a color (BGRA/BGR) or grayscale image.
    camera_intrinsic : numpy.ndarray
        The 3x3 intrinsic camera matrix.
    tags : list or None
        Optional precomputed detections from ``detect_apriltags_gray`` to avoid a second pass.

    Returns
    -------
    transform_mat : numpy.ndarray or None
        A 4x4 transformation matrix representing the rotation and translation,
        or None if insufficient valid tags are found or the PnP calculation fails.
    """

    if tags is None:
        _, tags = detect_apriltags_gray(observation, families=APRILTAG_FAMILY)
    print(f"Number of tags found: {len(tags)}")
    if tags:
        ids = sorted(set(int(t.tag_id) for t in tags))
        print(f"Detected tag ids: {ids}")
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

        _, tags = detect_apriltags_gray(cv_image, families=APRILTAG_FAMILY)
        vis = to_bgr_display(cv_image)
        draw_all_tag_overlays(vis, tags)

        t_cam_robot = get_transform_camera_robot(
            cv_image, camera_intrinsic, tags=tags
        )
        if t_cam_robot is not None:
            draw_pose_axes(vis, camera_intrinsic, t_cam_robot, size=TAG_SIZE)
            status = "PnP OK — pose axes drawn (playmat frame)"
            color = (0, 255, 0)
        else:
            status = "PnP FAILED — need enough corners from playmat ids 4-7 only"
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

        print(f"[checkpoint0] Preview: {len(tags)} tag(s). {status}")
        print("[checkpoint0] Press any key to close.")

        cv2.namedWindow("checkpoint0: AprilTag preview", cv2.WINDOW_NORMAL)
        cv2.imshow("checkpoint0: AprilTag preview", resize_for_preview(vis))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        zed.close()


if __name__ == "__main__":
    main()
