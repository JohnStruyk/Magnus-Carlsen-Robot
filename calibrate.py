import cv2
import numpy as np
from pupil_apriltags import Detector
from utils.zed_camera import ZedCamera

# --- 1. CONFIGURATION ---

# Robot Calibration
ROBOT_CALIB_CONFIG = {
    "tag_size": 0.08,
    "tag_ids": [0, 1, 2, 3],
    "tag_centers": {
        0: [0.38, 0.4], 1: [0.38, -0.4], 2: [0.0, 0.4], 3: [0.0, -0.4]
    }
}

# Chessboard Calibration
BOARD_CONFIG = {
    "tag_size": 0.08,        
    "square_size": 0.03,    
    "grid_size": (8, 8),     
    "tag_ids": [0, 1, 2, 3], # Adjust these to match your 5x5 tag IDs
    "tag_centers": {
        0: [0.0, 0.0],       # Origin: Bottom-Left Tag
        1: [0.0, 0.4],       # Bottom-Right
        2: [0.38, 0.0],      # Top-Left
        3: [0.38, 0.4]       # Top-Right
    },
    # Offset from Tag 0 center to the center of the first chessboard square (0,0)
    "grid_origin_offset": [0.05, -0.10] 
}

# --- 2. TRANSFORMATION LOGIC ---

def get_4x4_transform(tags, config, camera_intrinsic, strict=True):
    """
    Calculates the 4x4 transform matrix from the object to the camera.
    """
    target_ids = set(config["tag_ids"])
    found_tags = [t for t in tags if t.tag_id in target_ids]
    
    if strict and len(found_tags) < len(target_ids):
        return None, None, None
    if not strict and len(found_tags) == 0:
        return None, None, None

    world_points = []
    image_points = []
    half = config["tag_size"] / 2.0

    for tag in sorted(found_tags, key=lambda x: x.tag_id):
        cx, cy = config["tag_centers"][tag.tag_id]
        # 3D corners (Z=0): BL, BR, TR, TL
        wp_corners = [
            [cx - half, cy - half, 0], [cx - half, cy + half, 0], 
            [cx + half, cy + half, 0], [cx + half, cy - half, 0]  
        ]
        world_points.extend(wp_corners)
        image_points.extend(tag.corners)

    success, rvec, tvec = cv2.solvePnP(np.array(world_points, dtype=np.float32), 
                                      np.array(image_points, dtype=np.float32), 
                                      camera_intrinsic, None)
    if not success:
        return None, None, None

    rmat, _ = cv2.Rodrigues(rvec)
    t_mat = np.eye(4)
    t_mat[:3, :3] = rmat
    t_mat[:3, 3] = tvec.flatten()
    return t_mat, rvec, tvec

def get_board_centers_local(config):
    """Generates 8x8 grid centers as [X, Y, Z, 1] for matrix math."""
    grid_pts = []
    off_x, off_y = config["grid_origin_offset"]
    s = config["square_size"]

    for r in range(config["grid_size"][0]):
        for c in range(config["grid_size"][1]):
            # X decreases (Down), Y increases (Right) relative to origin
            grid_pts.append([off_x - (r * s), off_y + (c * s), 0.0, 1.0])
    return np.array(grid_pts)

# --- 3. MAIN EXECUTION ---

def main():
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic
    
    # Initialize detector for both 6x6 (36h11) and 5x5 (25h9) families
    detector = Detector(families='tag36h11 tag25h9')

    try:
        print("Grabbing frame... Ensure all tags are visible.")
        cv_image = zed.image
        if cv_image is None: return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2GRAY)
        tags = detector.detect(gray)

        # 1. Camera -> Robot Transform
        # solvePnP gives T_robot_to_cam, we invert it for T_cam_to_robot
        t_robot_to_cam, _, _ = get_4x4_transform(tags, ROBOT_CALIB_CONFIG, camera_intrinsic, strict=False)
        if t_robot_to_cam is None:
            print("Robot calibration tags not found.")
            return
        t_cam_to_robot = np.linalg.inv(t_robot_to_cam)

        # 2. Board -> Camera Transform
        t_board_to_cam, b_rvec, b_tvec = get_4x4_transform(tags, BOARD_CONFIG, camera_intrinsic, strict=True)
        if t_board_to_cam is None:
            print("Chessboard tags (5x5) not found or incomplete.")
            return

        # 3. Compute Centers in Robot Frame
        local_centers = get_board_centers_local(BOARD_CONFIG)
        robot_frame_centers = {}

        for i, p_local in enumerate(local_centers):
            # Transform: Board -> Camera -> Robot
            p_robot = t_cam_to_robot @ (t_board_to_cam @ p_local)
            robot_frame_centers[i] = p_robot[:3].tolist() # Store as [X, Y, Z]

        # 4. Visualization
        # Project 3D board points to 2D pixels
        pts_3d_xyz = local_centers[:, :3]
        grid_2d, _ = cv2.projectPoints(pts_3d_xyz, b_rvec, b_tvec, camera_intrinsic, None)

        for pt in grid_2d:
            cv2.circle(cv_image, tuple(pt.ravel().astype(int)), 4, (0, 255, 0), -1)

        for t in tags:
            # Draw tag boundaries
            pts = t.corners.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(cv_image, [pts], True, (255, 0, 0), 2)
            cv2.putText(cv_image, f"ID:{t.tag_id}", tuple(pts[0][0]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        print(f"Success! Square 0 in Robot Frame: {robot_frame_centers[0]}")
        cv2.imshow('Robot Frame Calibration', cv_image)
        cv2.waitKey(0)

        return robot_frame_centers

    finally:
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()