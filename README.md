# Magnus Carlsen Robot

Computer-vision + chess-engine + xArm integration for a physical chess robot.

## Repository structure

- `game_loop.py`: Main runtime loop (vision diff -> move validation -> robot reply).
- `game_persistence.py`: Saved-game I/O and startup FEN inference helpers.
- `move_diagnostics.py`: Move/castle inference + illegal-move diagnostics.
- `move_patterns.py`: Structured board-change classification helpers.
- `robot_turn.py`: Robot move execution/retry and engine-reply orchestration.
- `turn_processor.py`: Applies detected board changes to chess state with diagnostics.
- `ui_output.py`: Terminal output helpers (game-over banner and UI text primitives).
- `pickup_board_piece.py`: Motion planner/executor for pick, place, capture, promotion.
- `piece_continuity.py`: Board warp, tag-based board transform, occupancy comparison.
- `vision_debug.py`: Standalone visual debug runner for board-state detection.
- `calibrate_tags.py`: Camera/robot calibration and tag debugging.
- `stockfish_int.py`: Stockfish integration and optional SVG board rendering.
- `chess_utils.py`: Board-diff to candidate move conversion helpers.
- `camera_setup.py`: Live tag-family diagnostics.
- `manipulation_test.py`: Repeatable pick/place stress-test script.
- `pickup_cli.py`: One-off pick / capture from the command line.
- `utils/zed_camera.py`: ZED camera wrapper.
- `utils/vis_utils.py`: Visualization helpers.

## Runtime flow

1. Capture frame from ZED camera.
2. Detect board + piece state from AprilTags/warped board image.
3. Compare with previous board state to infer a candidate move.
4. Validate move legality with `python-chess`.
5. **Human plays White** — legal White moves are recorded on the internal board only; the arm never moves White pieces.
6. **Robot plays Black** — after each recorded White move, Stockfish chooses Black’s reply and the arm executes only that Black move.
7. Refresh the vision baseline after the robot moves so the next diff is human-only.

## Configuration hotspots

- **Board geometry / targeting:** `piece_continuity.py` (`BOARD_CONFIG` / `square_size`), `pickup_board_piece.py` (`HAND_EYE_XYZ_BIAS_M`, grasp offsets)
- **Motion safety + approach:** `pickup_board_piece.py`
  - `SAFE_Z`, `MIN_TOOL_Z_M`
  - `FORWARD_ENTRY_BOARD_FRACTION`, `MAX_FORWARD_ENTRY_STEP_M`
  - per-piece `GRASP_Z_OFFSET_*_M`
- **Game initialization:** `game_loop.py`
  - `CURRENT_FEN`
  - `CAPTURE_INTERVAL`
- **Stockfish engine path:** `stockfish_int.py`
  - `STOCKFISH_PATH` environment variable (preferred)
  - fallback `DEFAULT_STOCKFISH_PATH`

## Setup

Install dependencies in your Python environment:

- `opencv-python`
- `numpy`
- `python-chess`
- `pupil-apriltags`
- `scipy`
- `xarm-python-sdk`
- `pyzed` (ZED SDK Python bindings)

Install Stockfish binary and set:

- `export STOCKFISH_PATH=/absolute/path/to/stockfish`

## Common entry points

- Run full game loop:
  - `python game_loop.py`
- Run camera/tag diagnostics:
  - `python camera_setup.py`
- Run calibration helper:
  - `python calibrate_tags.py`
- Run manipulation stress test:
  - `python manipulation_test.py`
- One-off pick or capture (same motion stack as the game loop):
  - `python pickup_cli.py --piece-type pawn --from-square e2 --to-square e4`
  - Add `--captured-piece-type` for a capture sequence.
- Debug vision / move diffs (press `k` to quit):
  - `python vision_debug.py`

## Notes

- Saved game state is stored in `stored_game.txt`.

## Dependencies (from `dev`)

Files and packages the runtime depends on, including common transitive local modules.

### Local files

| File | Role |
|------|------|
| `piece_continuity.py` | Board state detection, warping, piece detection, board comparison |
| `chess_utils.py` | UCI move determination from board diff |
| `stockfish_int.py` | Stockfish engine interface, board SVG visualisation |
| `pickup_board_piece.py` | Robot arm pick-and-place, capture logic |
| `calibrate_tags.py` | AprilTag detection, camera-to-robot transform (used by `pickup_board_piece`) |
| `utils/zed_camera.py` | ZED camera wrapper (threaded image + point cloud capture) |
| `utils/vis_utils.py` | Pose axis drawing helpers (used by `calibrate_tags`) |

### Third-party packages

| Package | Purpose |
|---------|---------|
| `opencv-python` (`cv2`) | Image processing and display |
| `numpy` | Array operations |
| `python-chess` (`chess`, `chess.svg`, `chess.engine`) | Chess logic, FEN handling, SVG board rendering |
| `pupil-apriltags` | AprilTag detection for board and playmat localisation |
| `scipy` | Rotation utilities in pick-and-place (`scipy.spatial.transform`) |
| `xarm-python-sdk` (`xarm.wrapper`) | Robot arm control |
| `pyzed` | ZED SDK Python bindings |
| `stockfish` (binary) | Chess engine — path configured via `STOCKFISH_PATH` / `stockfish_int.py` |

Legacy scripts and assets from `dev` live under `old_code/`.
