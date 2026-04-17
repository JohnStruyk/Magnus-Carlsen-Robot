import cv2, numpy
from pupil_apriltags import Detector

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera

TAG_SIZE = 0.08
APRILTAG_FAMILY = "tag36h11"
PREVIEW_MAX_WIDTH = 1280

# top-left, top-right, bottom-left, bottom-right
TAG_CENTER_COORDINATES = [[0.38, 0.4],
                         [0.38, -0.4],
                         [0.0, 0.4],
                         [0.0, -0.4]]

def get_pnp_pairs(tags):
    """
    Extract corresponding 3D world coordinates and 2D image coordinates for 
    the corners of detected AprilTags.

    This function iterates through the detected tags, filters for specific tag IDs 
    (0 through 3), and computes the 3D world coordinates of their four corners 
    based on predefined center coordinates and tag size. It maps these to the 
    corresponding 2D pixel coordinates found in the image.

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
        if tid < 0 or tid > 3:
            continue
        tag_center = TAG_CENTER_COORDINATES[tid]

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
    n_playmat = sum(1 for t in tags if 0 <= int(t.tag_id) <= 3)
    summary = f"tags={len(tags)} (ids 0-3 count={n_playmat}) family={APRILTAG_FAMILY}"
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


def partition_playmat_and_board_tags(tags, board_tag_ids=(4, 5, 6, 7)):
    """
    Split detections into four playmat tags (ids 0–3) and four board tags.

    - If board corners use IDs 4–7, playmat = {0,1,2,3}, board = {4,5,6,7}.
    - Else if board reuses 0–3, expect two detections per ID (mat + board); assign the
      larger polygon in image space to the playmat (usually closer / larger on ZED).

    Returns (playmat_tags, board_tags, debug_msg). On failure, tags are None and msg explains.
    """
    by_id = {}
    for t in tags:
        tid = int(t.tag_id)
        by_id.setdefault(tid, []).append(t)

    board_ids = list(board_tag_ids)

    # Case 1: distinct board IDs (e.g. 4–7)
    if all(bid in by_id and len(by_id[bid]) >= 1 for bid in board_ids):
        playmat = []
        board = []
        for i in range(4):
            if i not in by_id:
                return None, None, f"playmat: missing tag id {i}"
            cands = sorted(by_id[i], key=_tag_polygon_area_sq_px, reverse=True)
            playmat.append(cands[0])
        for bid in board_ids:
            cands = sorted(by_id[bid], key=_tag_polygon_area_sq_px, reverse=True)
            board.append(cands[0])
        return playmat, board, "split: playmat 0-3, board 4-7"

    # Case 2: duplicate 0–3 on mat and board
    playmat = []
    board = []
    for i in range(4):
        if i not in by_id:
            return None, None, f"missing tag id {i} (need 0-3 on playmat, and board tags or duplicates)"
        cands = sorted(by_id[i], key=_tag_polygon_area_sq_px, reverse=True)
        playmat.append(cands[0])
        if len(cands) < 2:
            return (
                None,
                None,
                f"tag id {i}: only one detection; use board tags {board_ids} or print duplicate 0-3 on board",
            )
        board.append(cands[1])

    return playmat, board, "split: duplicate 0-3 by larger area -> playmat, smaller -> board"


def get_transform_camera_robot_from_tags(tags, camera_intrinsic):
    """
    Same as get_transform_camera_robot but uses an existing tag list (no second detect).
    Pass exactly the four playmat tags (ids 0–3, one each) so board duplicates are excluded.
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
            status = "PnP FAILED — need enough corners from playmat ids 0-3 only"
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
