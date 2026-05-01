# Magnus Carlsen Robot

Vision + Stockfish + xArm for physical chess. **Human = White**, **robot = Black** only.

## Layout

| Module | Role |
|--------|------|
| `game_loop.py` | Main loop, game-over banner, save on exit |
| `turn_processor.py` | Board diff → move string, `BoardChange`, legality, apply move (includes en passant / castle inference) |
| `robot_turn.py` | Stockfish + arm (Black only) |
| `pickup_board_piece.py` | Pick/place/capture/promotion |
| `piece_continuity.py` | Tags, warp, occupancy |
| `calibrate_tags.py` | Tag preview + PnP; `ROBOT_IP_DEFAULT` |
| `game_persistence.py` | Saved game / FEN startup |
| `stockfish_int.py` | Engine + SVG board |
| `pickup_cli.py` | One-off `move_piece` / `capture_piece` |
| `utils/zed_camera.py` | ZED RGB + intrinsics |

## Run

```bash
export STOCKFISH_PATH=/path/to/stockfish
python game_loop.py
python calibrate_tags.py
python pickup_cli.py --piece-type pawn --from-square e2 --to-square e4
```

Config: `game_loop.py` (`CURRENT_FEN`, `CAPTURE_INTERVAL`), `piece_continuity.py` / `pickup_board_piece.py` (geometry & motion), `stockfish_int.py` (engine path).

Saved state: `stored_game.txt`.

## Dependencies

`opencv-python`, `numpy`, `python-chess`, `pupil-apriltags`, `scipy`, `xarm-python-sdk`, `pyzed`, Stockfish binary.
