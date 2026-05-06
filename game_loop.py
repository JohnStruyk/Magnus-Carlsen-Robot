"""Main entry: grab frames, diff occupancy against python-chess (you're White, arm is Black)."""

import time
import faulthandler
from typing import Callable, Optional

import chess
from pupil_apriltags import Detector

from utils.zed_camera import ZedCamera
from piece_continuity import get_board_state, compare_board_states
from game_persistence import detect_starting_fen, prompt_continue_saved_game, save_game_state
from robot_turn import execute_robot_reply_turn
from turn_processor import BoardChange, process_detected_change

CAPTURE_INTERVAL = 5
CURRENT_FEN = ""


class LoopState:
    def __init__(self):
        self.chess_board: Optional[chess.Board] = None
        self.prior_board_state: Optional[object] = None
        self.resume_robot_pending = False


def print_game_over_banner(chess_board: chess.Board) -> None:
    oc = chess_board.outcome(claim_draw=True)
    term = oc.termination.name.replace("_", " ") if oc else "unknown"  # mate, stalemate, ...
    res = oc.result() if oc else chess_board.result(claim_draw=True)  # 1-0 / 0-1 / 1/2-1/2
    print("\n=== GAME OVER ===")
    print(f"Result: {res} | {term}")
    print(f"FEN: {chess_board.fen()}\n")


def build_try_robot_reply(
    state: LoopState,
    zed: ZedCamera,
    detector,
    camera_intrinsic,
    print_banner: Callable[[chess.Board], None],
) -> Callable[[str], bool]:
    """Closes over ``state``. Runs the arm path Stockfish chose for Black."""

    def try_robot_reply(error_prefix: str) -> bool:
        # True means stop the outer loop (mate/stalemate/etc. after robot moved).
        if state.chess_board is None or state.chess_board.turn != chess.BLACK:
            print(f"  {error_prefix}: skipped (not Black to move, robot never plays White).")
            return False
        try:
            _, state.prior_board_state, game_over = execute_robot_reply_turn(
                state.chess_board, zed, detector, camera_intrinsic, print_banner
            )
            return game_over
        except Exception as exc:
            print(f"  {error_prefix}: {exc}")
            return False

    return try_robot_reply


def try_initialize_board(state: LoopState, board_state, current_fen: str) -> bool:
    """First good frame builds ``chess_board``. Returns False to bail on unknown layout."""
    if state.chess_board is not None:
        return True
    fen = detect_starting_fen(board_state, current_fen)
    if fen is None:
        print("WARNING: Unknown board state. Set CURRENT_FEN at the top of game_loop.py and restart.")
        return False
    state.chess_board = chess.Board(fen)
    print(f"Chess board initialized from FEN: {fen}")
    return True


def run_change_branch(
    state: LoopState,
    board_state,
    try_robot_reply: Callable[[str], bool],
) -> bool:
    """After a baseline exists: diff, legalize, maybe run robot. True = exit main."""
    # Black markers then white: removals/additions from vision compare_board_states().
    one_r, two_r, one_a, two_a = compare_board_states(state.prior_board_state, board_state)
    change = BoardChange(one_r, two_r, one_a, two_a)

    if change.changed:
        result = process_detected_change(
            state.chess_board,
            state.chess_board.turn,
            board_state,
            change,
            try_robot_reply,
            print_game_over_banner,
        )
        state.prior_board_state = result.prior_board_state
        return result.game_over

    # Board unchanged but Black still owes a move (for example resume after failed arm attempt).
    if state.chess_board.turn == chess.BLACK:
        game_over = try_robot_reply("Black (robot) move failed")
        state.resume_robot_pending = False
        return game_over
    print("No change detected.")
    return False


def run_no_prior_branch(state: LoopState, board_state, resumed: bool) -> None:
    """Stores first grid snapshot. ``resume_robot_pending`` suppresses noisy retries until baseline exists."""
    state.prior_board_state = board_state
    if resumed and state.resume_robot_pending:
        print("Resume baseline captured. Waiting for stable frame before robot move.")
    print("No prior state to compare against.")


def main() -> None:
    faulthandler.enable(all_threads=True)
    zed = ZedCamera()
    detector = Detector(families="tag36h11 tag25h9")
    camera_intrinsic = zed.camera_intrinsic

    state = LoopState()
    state.chess_board, resumed = prompt_continue_saved_game()
    state.resume_robot_pending = bool(
        resumed and state.chess_board is not None and state.chess_board.turn == chess.BLACK
    )

    try_robot_reply = build_try_robot_reply(state, zed, detector, camera_intrinsic, print_game_over_banner)

    try:
        for i in range(500):
            print(f"\n--- Iteration {i + 1}/500 ---")
            loop_start = time.time()

            cv_image = zed.image
            board_state, _, _ = get_board_state(cv_image, detector, camera_intrinsic)

            if board_state is None:
                print("Could not detect board this iteration, skipping.")
            else:
                if not try_initialize_board(state, board_state, CURRENT_FEN):
                    break

                if state.prior_board_state is not None:
                    if run_change_branch(state, board_state, try_robot_reply):
                        break
                else:
                    run_no_prior_branch(state, board_state, resumed)

            elapsed = time.time() - loop_start
            sleep_time = max(0, CAPTURE_INTERVAL - elapsed)  # pad iteration to CAPTURE_INTERVAL seconds
            print(f"Sleeping {sleep_time:.1f}s until next capture.")
            time.sleep(sleep_time)

    finally:
        save_game_state(state.chess_board)
        zed.close()


if __name__ == "__main__":
    main()
