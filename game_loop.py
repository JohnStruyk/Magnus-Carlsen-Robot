import time
import faulthandler
import chess
from pupil_apriltags import Detector

from utils.zed_camera import ZedCamera
from piece_continuity import get_board_state, compare_board_states
from game_persistence import detect_starting_fen, prompt_continue_saved_game, save_game_state
from move_patterns import build_board_change
from robot_turn import execute_robot_reply_turn
from turn_processor import process_detected_change
from ui_output import print_game_over_banner

CAPTURE_INTERVAL = 5  # seconds between captures

# Fallback FEN if board is not in standard starting position.
# Fill this in if you are starting from a known non-standard position.
CURRENT_FEN = ""

# Human plays White; the arm only executes Black (Stockfish). ``chess_board.turn`` is the source of truth.
_chess_board_ref = [None]


def main():
    faulthandler.enable(all_threads=True)
    zed = ZedCamera()
    detector = Detector(families='tag36h11 tag25h9')
    camera_intrinsic = zed.camera_intrinsic

    prior_board_state = None

    # --- Saved game check ---
    chess_board, resumed = prompt_continue_saved_game()
    resume_robot_pending = bool(
        resumed and chess_board is not None and chess_board.turn == chess.BLACK
    )
    _chess_board_ref[0] = chess_board  # keep signal handler in sync

    def try_robot_reply(error_prefix: str):
        nonlocal prior_board_state, chess_board
        if chess_board is None or chess_board.turn != chess.BLACK:
            print(f"  {error_prefix}: skipped — not Black to move (robot never plays White).")
            return False
        try:
            _, prior_board_state, game_over = execute_robot_reply_turn(
                chess_board, zed, detector, camera_intrinsic, print_game_over_banner
            )
            _chess_board_ref[0] = chess_board
            return game_over
        except Exception as exc:
            print(f"  {error_prefix}: {exc}")
            return False

    try:
        for i in range(500):
            print(f"\n--- Iteration {i + 1}/500 ---")
            loop_start = time.time()

            cv_image = zed.image
            board_state, _, _ = get_board_state(cv_image, detector, camera_intrinsic)

            if board_state is None:
                print("Could not detect board this iteration, skipping.")
            else:
                # Initialize chess board from first good detection (skip if resumed from save)
                if chess_board is None:
                    fen = detect_starting_fen(board_state, CURRENT_FEN)
                    if fen is None:
                        print("WARNING: Unknown board state. Set CURRENT_FEN at the top of game_loop.py and restart.")
                        break
                    chess_board = chess.Board(fen)
                    _chess_board_ref[0] = chess_board
                    print(f"Chess board initialized from FEN: {fen}")

                if prior_board_state is not None:
                    one_removals, two_removals, one_additions, two_additions = compare_board_states(prior_board_state, board_state)
                    change = build_board_change(one_removals, two_removals, one_additions, two_additions)

                    if change.changed:
                        result = process_detected_change(
                            chess_board,
                            chess_board.turn,
                            board_state,
                            change,
                            try_robot_reply,
                            print_game_over_banner,
                        )
                        prior_board_state = result.prior_board_state
                        _chess_board_ref[0] = chess_board
                        if result.game_over:
                            break
                    else:
                        # Black to move and board unchanged: play or retry Stockfish/arm (e.g. resume or failed attempt).
                        if chess_board.turn == chess.BLACK:
                            game_over = try_robot_reply("Black (robot) move failed")
                            resume_robot_pending = False
                            if game_over:
                                break
                        else:
                            print("No change detected.")
                else:
                    prior_board_state = board_state
                    if resumed and resume_robot_pending:
                        print("Resume baseline captured. Waiting for stable frame before robot move.")
                    print("No prior state to compare against.")

            elapsed = time.time() - loop_start
            sleep_time = max(0, CAPTURE_INTERVAL - elapsed)
            print(f"Sleeping {sleep_time:.1f}s until next capture.")
            time.sleep(sleep_time)

    finally:
        # Save game state on any exit (normal finish, exception, etc.)
        save_game_state(_chess_board_ref[0])


if __name__ == "__main__":
    main()
