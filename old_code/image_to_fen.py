import cv2
import numpy as np

def chessboard_corners_to_quad(corners, pattern_size):

    print(corners.shape)
    corners = corners.reshape(-1, 2)
    print(corners.shape)

    w, h = pattern_size

    top_left = corners[0]
    top_right = corners[w - 1]
    bottom_left = corners[(h - 1) * w]
    bottom_right = corners[-1]

    return np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32)

def warp_chessboard_from_corners(image_bgr, quad, pattern_size=(7, 7)):
    """
    Warp a chessboard to a top-down view using detected inner corners.

    Parameters
    ----------
    image_bgr : np.ndarray
    quad : np.ndarray (4x2) from chessboard_corners_to_quad
    pattern_size : (cols, rows) inner corners (e.g., (7,7) for 8x8 board)

    Returns
    -------
    warped : np.ndarray
    """

    # unpack quad (already ordered: TL, TR, BR, BL)
    tl, tr, br, bl = quad.astype(np.float32)

    # estimate square size using inner grid spacing
    board_cols, board_rows = pattern_size

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    avg_width = (width_top + width_bottom) / 2.0

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    avg_height = (height_left + height_right) / 2.0

    # estimate size of one square
    square_w = avg_width / (board_cols - 1)
    square_h = avg_height / (board_rows - 1)
    square_size = (square_w + square_h) / 2.0

    # total board size (add 1 because inner corners = squares - 1)
    out_cols = board_cols + 1
    out_rows = board_rows + 1

    out_width = int(square_size * out_cols)
    out_height = int(square_size * out_rows)

    # destination points (perfect rectangle)
    dst = np.array([
        [0, 0],
        [out_width - 1, 0],
        [out_width - 1, out_height - 1],
        [0, out_height - 1]
    ], dtype=np.float32)

    src = np.array([tl, tr, br, bl], dtype=np.float32)

    # perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image_bgr, M, (out_width, out_height))

    cv2.imshow("Warped Chessboard", warped)
    cv2.waitKey(0)

    return warped

def find_chessboard_with_pattern(img, pattern_size=(7, 7)):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
    ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)

    if ret:
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)

    cv2.imshow("Chessboard Detection", img)
    cv2.waitKey(0)

    return corners

img = cv2.imread("wooden_sample.jpg")

img = cv2.imread("easy_sample.jpeg")

img = cv2.imread("sample_chess.jpg")

img = cv2.imread("more_chesss.jpg")



corners = find_chessboard_with_pattern(img)

quad = chessboard_corners_to_quad(corners, (7, 7))

warp_chessboard_from_corners(img, quad, (7, 7))

#print(corners)