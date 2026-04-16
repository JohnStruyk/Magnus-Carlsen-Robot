import cv2
import numpy as np
from pupil_apriltags import Detector
from utils.zed_camera import ZedCamera

# --- 1. CONFIGURATION ---

# Robot Calibration (4 tags on robot mat to establish robot-to-camera)
ROBOT_CALIB_CONFIG = {
    "tag_size": 0.08,
    "tag_ids": [0, 1, 2, 3],
    "tag_centers": {
        0: [0.38, 0.4], 
        1: [0.38, -0.4], 
        2: [0.0, 0.4], 
        3: [0.0, -0.4]
    }
}

# Chessboard Calibration (4 tags on board to establish board-to-camera)
BOARD_CONFIG = {
    "tag_size": 0.0265,        
    "square_size": 0.025,    
    "grid_size": (8, 8),     
    "tag_ids": [0, 1, 2, 3], # Adjust these to match your 5x5 tag IDs
    "tag_centers": {
        0: [0.0, 0.0],       # Origin: Bottom-Left Tag
        1: [0.0, 0.15572],       # Top Left
        2: [0.23713, 0.0],      # Bottom-Right
        3: [0.23713, 0.15572]       # Top-Right
    },
    # Offset from Tag 0 center to the center of the first chessboard square (0,0)
    #"grid_origin_offset": [0.03, -0.008] 
    "grid_origin_offset": [0.0175, -0.0205] 
}

TAG_WIDTH_OFFSET = 17.5
TAG_HEIGHT_OFFSET = -20.5




# --- 2. TRANSFORMATION LOGIC ---

def get_4x4_transform(tags, config, camera_intrinsic, strict=True):
    """Calculates the 4x4 transform matrix from the object to the camera."""
    target_ids = set(config["tag_ids"])
    found_tags = [t for t in tags if t.tag_id in target_ids]
    
    # Logic to specify what should happen if not all the tags are seen, want all 4 for the chessboard part
    if strict and len(found_tags) < len(target_ids):
        return None, None, None
    if not strict and len(found_tags) == 0:
        return None, None, None

    world_points = []
    image_points = []
    half = config["tag_size"] / 2.0

    # Sort tags to ensure consistent mapping
    for tag in sorted(found_tags, key=lambda x: x.tag_id):
        cx, cy = config["tag_centers"][tag.tag_id]
        # World points for tag corners: BL, BR, TR, TL
        wp_corners = [
            [cx - half, cy - half, 0], 
            [cx - half, cy + half, 0], 
            [cx + half, cy + half, 0], 
            [cx + half, cy - half, 0]  
        ]
        world_points.extend(wp_corners)
        image_points.extend(tag.corners)

    success, rvec, tvec = cv2.solvePnP(
        np.array(world_points, dtype=np.float32), 
        np.array(image_points, dtype=np.float32), 
        camera_intrinsic, None
    )
    
    if not success:
        return None, None, None

    rmat, _ = cv2.Rodrigues(rvec)
    t_mat = np.eye(4)
    t_mat[:3, :3] = rmat
    t_mat[:3, 3] = tvec.flatten()
    return t_mat, rvec, tvec 

def get_board_centers_local(config):
    """Generates 3D coordinates for the grid [X, Y, Z, 1]."""
    grid_pts = []
    off_x, off_y = config["grid_origin_offset"]
    s = config["square_size"]

    for r in range(config["grid_size"][0]):
        for c in range(config["grid_size"][1]):
            # +X is Up (rows), +Y is Right (cols)
            grid_pts.append([off_x + (r * s), off_y + (c * s), 0.0, 1.0])
    return np.array(grid_pts, dtype=np.float32)

def get_warped(img, b_rvec, b_tvec, intrix, square_px):
    W = BOARD_CONFIG["grid_size"][1] * square_px
    H = BOARD_CONFIG["grid_size"][0] * square_px

    off_x, off_y = BOARD_CONFIG["grid_origin_offset"]

    
    board_corners_3d = np.array([
        BOARD_CONFIG["tag_centers"][0],  # bottom-left
        BOARD_CONFIG["tag_centers"][1],  # top-left
        BOARD_CONFIG["tag_centers"][3],  # top-right
        BOARD_CONFIG["tag_centers"][2],  # bottom-right
    ], dtype=np.float32)

    board_corners_3d[0, 0] += off_x
    board_corners_3d[0, 1] += off_y

    board_corners_3d[1, 0] += off_x
    board_corners_3d[1, 1] -= off_y

    board_corners_3d[2, 0] -= off_x
    board_corners_3d[2, 1] -= off_y

    board_corners_3d[3, 0] -= off_x
    board_corners_3d[3, 1] += off_y

    # add z=0
    board_corners_3d = np.hstack([board_corners_3d, np.zeros((4,1), dtype=np.float32)])
    

    img_corners, _ = cv2.projectPoints(
        board_corners_3d,
        b_rvec,
        b_tvec,
        intrix,
        None
    )

    img_corners = img_corners.reshape(-1, 2)

    '''FINDING TAGS USING APRIL TAG NOT BOARD CONFIG
    tag_dict = {t.tag_id: t for t in tags if t.tag_id in BOARD_CONFIG["tag_ids"]}

    # Ensure all 4 tags exist
    if not all(tid in tag_dict for tid in BOARD_CONFIG["tag_ids"]):
        print("Not all board tags detected.")
        return

    # Order: BL, TL, TR, BR (must match dst_corners!)
    img_corners = np.array([
        tag_dict[0].center,  # bottom-left
        tag_dict[1].center,  # top-left
        tag_dict[3].center,  # top-right
        tag_dict[2].center,  # bottom-right
    ], dtype=np.float32)
    '''
    

    dst_corners = np.array([
        [0, H],
        [0, 0],
        [W, 0],
        [W, H]
    ], dtype=np.float32)

    H_mat, _ = cv2.findHomography(img_corners, dst_corners)

    warped = cv2.warpPerspective(img, H_mat, (W, H))

    #img corners are in 2d image space i think
    return warped, img_corners

def detect_pieces(warped, square_px):
    board_state = np.zeros((8,8),dtype=int)

    for row in range(8):
        for col in range(8):
            cx = int(col*square_px+square_px/2)
            cy = int(row*square_px+square_px/2)

            region_size = 20 // 2 #pixel size of window
            region = warped[cy-region_size:cy+region_size, cx-region_size:cx+region_size]
            avg_color = region.mean(axis=(0,1))

            b,g,r = avg_color[:3]
            total = b+g+r

            #classify by color
            if g > r +30 and b > r+30 and total>130: #green piece, shows up turquoise ish
                board_state[row,col] = 1
            elif r > b +30 and g > b+30 and total<400: #yellow
                board_state[row,col] = 2
            elif r > b+30 and r > g+30: #red
                board_state[row,col] = 2
            
    return board_state


def draw_piece_detected(warped, board_state, square_px):
    overlay = warped.copy()

    for r in range(8):
        for c in range(8):

            cx = int(c*square_px+square_px/2)
            cy = int(r*square_px+square_px/2)

            val = board_state[r,c]

            if val == 1: #we found a green piece
                cv2.circle(overlay, (cx,cy), 12, (255,0,255),-1)

            elif val == 2: #red/yellow
                cv2.circle(overlay, (cx,cy), 12, (255,0,0),-1)

            else:
                cv2.rectangle(overlay, (cx-10,cy-10), (cx+10,cy+10),(0,0,255),3 )
    
    return overlay

def compare_board_states(old_state, new_state):

    one_removals = np.argwhere((old_state == 1) & (new_state != 1))
    one_additions = np.argwhere((old_state != 1) & (new_state == 1))

    two_removals = np.argwhere((old_state == 2) & (new_state != 2))
    two_additions = np.argwhere((old_state != 2) & (new_state == 2))

    return one_removals, two_removals, one_additions, two_additions

def determine_move(one_removals, two_removals, one_additions, two_additions):

    if len(one_additions) + len(two_additions) > len(one_removals) + len(two_removals):
        return "BAD. there are more pieces now than at start of move"

    if len(one_additions) + len(two_additions) + 1 < len(one_removals) + len(two_removals):
        return "BAD. too many pieces removed"

    if len(one_additions) > 0 and len(two_additions) > 0:
        return "BAD. both color pieces have moved."

    if len(one_removals) + len(two_removals) == 0:
        return "BAD. neither color piece has moved."

    if len(one_removals) == 1 and len(one_additions) == 1:
        if len(two_removals) == 1:
            return "FINE. team one captured a piece."
        else:
            return "FINE. team one made a normal move."

    if len(one_removals) == 2 and len(one_additions) == 2:
        return "FINE. team one tried to castle."

    if len(two_removals) == 1 and len(two_additions) == 1:
        if len(one_removals) == 1:
            return "FINE. team two captured a piece."
        else:
            return "FINE. team two made a normal move."

    if len(two_removals) == 2 and len(two_additions) == 2:
        return "FINE. team two tried to castle."



def get_board_state(cv_image, detector, camera_intrinsic):
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)

    # 1. Camera -> Robot Transform (Using original Robot Tag logic)
    t_robot_to_cam, _, _ = get_4x4_transform(tags, ROBOT_CALIB_CONFIG, camera_intrinsic, strict=False)
    if t_robot_to_cam is None:
        print("Robot tags (0-3) not found.")
        return
    t_cam_to_robot = np.linalg.inv(t_robot_to_cam)

    # 2. Board -> Camera Transform (Using your new Tag 4-7 logic)
    t_board_to_cam, b_rvec, b_tvec = get_4x4_transform(tags, BOARD_CONFIG, camera_intrinsic, strict=True)
    if t_board_to_cam is None:
        print("Chessboard tags (4-7) not found.")
        resized_img = cv2.resize(cv_image, (1080, 700))
        cv2.imshow('Robot Calibration', resized_img)        
        cv2.waitKey(0)
        return

    # 3. Compute Centers in Robot Frame
    local_centers = get_board_centers_local(BOARD_CONFIG)
    robot_frame_centers = {}

    for i, p_local in enumerate(local_centers):
        # Transformation: Board -> Camera -> Robot
        p_robot = t_cam_to_robot @ (t_board_to_cam @ p_local)
        robot_frame_centers[i] = p_robot[:3].tolist()

    # 4. Visualization
    pts_3d_xyz = local_centers[:, :3].astype(np.float32)
    grid_2d, _ = cv2.projectPoints(pts_3d_xyz, b_rvec, b_tvec, camera_intrinsic, None)

    for pt in grid_2d:
        cv2.circle(cv_image, tuple(pt.ravel().astype(int)), 4, (0, 255, 0), -1)

    # Add labels for debugging
    for t in tags:
        center = tuple(t.center.astype(int))
        cv2.putText(cv_image, f"ID:{t.tag_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    print(f"Captured {len(tags)} tags.")
    print(f"Square 0 (Robot Frame): {robot_frame_centers[0]}")

    output_square_px = 100

    warped, img_corners = get_warped(cv_image, b_rvec, b_tvec, camera_intrinsic, output_square_px)
    
    #Comment this out to remove piece detection
    board_state = detect_pieces(warped, output_square_px)
    warped_with_pieces = draw_piece_detected(warped, board_state, output_square_px)


    for pt in img_corners:
        cv2.circle(cv_image, tuple(pt.astype(int)), 10, (0,0,255), -1)

    resized_img = cv2.resize(cv_image, (1080, 700))

    overlay = np.zeros_like(warped)
    colors = [(128, 0, 128), (0, 165, 255)]  # Purple and Orange (BGR)

    for r in range(BOARD_CONFIG["grid_size"][0]):
        for c in range(BOARD_CONFIG["grid_size"][1]):
            color = colors[(r + c) % 2]
            top_left = (c * output_square_px, r * output_square_px)
            bottom_right = ((c + 1) * output_square_px, (r + 1) * output_square_px)
            cv2.rectangle(overlay, top_left, bottom_right, color, -1)

    # Blend: 0.5 (original) + 0.5 (overlay)
    #warped = cv2.addWeighted(overlay, 0.3, warped, 0.7, 0)

    cv2.imshow('Robot Calibration', resized_img)
    cv2.waitKey(0)

    #cv2.imshow("Warped Chessboard", warped)
    cv2.imshow('Warped with Piece Detection', warped_with_pieces)
    cv2.waitKey(0)

    return board_state


# --- 3. MAIN ---

def main():
    zed = ZedCamera()
   # camera_intrinsic = zed.camera_intrinsic
    detector = Detector(families='tag36h11 tag25h9')

    camera_intrinsic = np.array(((1062.18, 0, 1047.36), (0, 1062.18, 610.32), (0, 0, 1)))

    init_image = zed.image

    prior_board_state = get_board_state(init_image, detector, camera_intrinsic)

    try:
        while True:

            cv_image = zed.image

            board_state = get_board_state(cv_image, detector, camera_intrinsic) # These are the center locations of the chess board squares relative to the robot frame

            print(board_state)

            one_removals, two_removals, one_additions, two_additions = compare_board_states(prior_board_state, board_state)

            key_pressed = cv2.waitKey(0)
            if key_pressed == ord('k'):
                break

            cv2.destroyAllWindows()

            prior_board_state = board_state

            print(f"one_removals: {one_removals}")
            print(f"two_removals: {two_removals}")
            print(f"one_additions: {one_additions}")
            print(f"two_additions: {two_additions}")




    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
