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
- `utils/zed_camera.py`: ZED camera wrapper.
- `utils/vis_utils.py`: Visualization helpers.

## Runtime flow

1. Capture frame from ZED camera.
2. Detect board + piece state from AprilTags/warped board image.
3. Compare with previous board state to infer a candidate move.
4. Validate move legality with `python-chess`.
5. If legal human move: update board, ask Stockfish for reply, execute robot move.
6. Refresh visual baseline after robot move to avoid replaying robot actions.

## Configuration hotspots

- **Board geometry / targeting:** `pickup_board_piece.py`
  - `BOARD_TOTAL_SIZE_IN`
  - `CHESS_SQUARE_SIZE_M`
  - `HAND_EYE_XYZ_BIAS_M`
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

## Notes

- Saved game state is stored in `stored_game.txt`.
