# DEPENDENCIES

Files and packages that `game_loop.py` depends on, including transitive local dependencies.

## Local files

| File | Role |
|------|------|
| `piece_continuity.py` | Board state detection, warping, piece detection, board comparison |
| `chess_utils.py` | UCI move determination from board diff |
| `stockfish_int.py` | Stockfish engine interface, board SVG visualisation |
| `pickup_board_piece.py` | Robot arm pick-and-place, capture logic |
| `calibrate_tags.py` | AprilTag detection, camera-to-robot transform (used by `pickup_board_piece`) |
| `utils/zed_camera.py` | ZED camera wrapper (threaded image + point cloud capture) |
| `utils/vis_utils.py` | Pose axis drawing helpers (used by `pickup_board_piece` and `calibrate_tags`) |

## Third-party packages

| Package | Purpose |
|---------|---------|
| `opencv-python` (`cv2`) | Image processing and display |
| `numpy` | Array operations |
| `python-chess` (`chess`, `chess.svg`, `chess.engine`) | Chess logic, FEN handling, SVG board rendering |
| `pupil-apriltags` | AprilTag detection for board and playmat localisation |
| `scipy` | Rotation utilities in pick-and-place (`scipy.spatial.transform`) |
| `xarm-python-sdk` (`xarm.wrapper`) | Robot arm control |
| `pyzed` | ZED SDK Python bindings |
| `stockfish` (binary) | Chess engine — path configured in `stockfish_int.py` |
