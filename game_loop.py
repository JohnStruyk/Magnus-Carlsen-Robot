import cv2
import time
import signal
import sys
import os
import faulthandler
import numpy as np
import chess
import chess.svg
import webbrowser
from pupil_apriltags import Detector

from utils.zed_camera import ZedCamera
from piece_continuity import get_board_state, display_board_state, compare_board_states
import chess_utils
from stockfish_int import get_best_move, visualize_board
from pickup_board_piece import move_piece, capture_piece

CAPTURE_INTERVAL = 5  # seconds between captures
SAVED_GAME_FILE = "stored_game.txt"

# Fallback FEN if board is not in standard starting position.
# Fill this in if you are starting from a known non-standard position.
CURRENT_FEN = ""

STANDARD_STARTING_FEN = chess.STARTING_FEN

# Standard starting board_state layout matching our camera orientation:
# row 0 = rank 8 (black back rank), row 7 = rank 1 (white back rank)
# 1 = green = black, 2 = yellow/red = white
STANDARD_BOARD_STATE = np.array([
    [1,1,1,1,1,1,1,1],  # row 0: rank 8 - black back rank
    [1,1,1,1,1,1,1,1],  # row 1: rank 7 - black pawns
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [2,2,2,2,2,2,2,2],  # row 6: rank 2 - white pawns
    [2,2,2,2,2,2,2,2],  # row 7: rank 1 - white back rank
], dtype=int)


def row_col_to_chess_square(row, col):
    """Convert board_state (row, col) to a python-chess Square."""
    return chess.square(col, 7 - row)


def detect_starting_fen(board_state):
    """
    If board has 32 pieces in standard positions, return standard FEN.
    If CURRENT_FEN is set, return that. Otherwise return None.
    """

    print(board_state)
    print(STANDARD_BOARD_STATE)

    if np.array_equal(board_state, STANDARD_BOARD_STATE):
        print("Standard starting position detected.")
        return STANDARD_STARTING_FEN

    if CURRENT_FEN:
        print("Using CURRENT_FEN.")
        return CURRENT_FEN

    return None

def count_white_captures(chess_board):
    """
    Returns how many white pieces have been captured.
    Assumes standard starting count (16 pieces) and ignores promotions.
    """
    white_pieces_on_board = sum(1 for piece in chess_board.piece_map().values() if piece.color == chess.WHITE)
    
    return 16 - white_pieces_on_board


def describe_move(chess_board, removals, additions):
    """
    Given the chess.Board before the move, and the removal/addition arrays
    from compare_board_states, print the piece, color, from-square, and to-square.
    """
    # Flatten removals/additions across both colors into (square, color_val) pairs
    # color_val: 1=black(green), 2=white(yellow/red)
    all_removals = [(tuple(sq), 1) for sq in removals[0]] + [(tuple(sq), 2) for sq in removals[1]]
    all_additions = [(tuple(sq), 1) for sq in additions[0]] + [(tuple(sq), 2) for sq in additions[1]]

    # The moving piece is the one that appears in both removals and additions with the same color
    for (from_rc, from_color) in all_removals:
        for (to_rc, to_color) in all_additions:
            if from_color == to_color:
                from_sq = row_col_to_chess_square(*from_rc)
                to_sq = row_col_to_chess_square(*to_rc)

                piece = chess_board.piece_at(from_sq)
                color_name = "black" if from_color == 1 else "white"
                piece_name = chess.piece_name(piece.piece_type) if piece else "unknown"
                from_alg = chess.square_name(from_sq)
                to_alg = chess.square_name(to_sq)

                print(f"  {color_name} {piece_name} moved from {from_alg} to {to_alg}")
                return

    print("  Could not identify moving piece.")

def parse_move_string(chess_board, move_string):
    from_square = move_string[:2]
    to_square = move_string[2:]
    from_occupant = chess_board.piece_at(chess.parse_square(from_square))
    to_occupant = chess_board.piece_at(chess.parse_square(to_square))
    from_occupant = from_occupant.symbol()
    if to_occupant is not None:
        to_occupant = to_occupant.symbol()
    
    return from_square, to_square, from_occupant, to_occupant


def print_game_over_banner(chess_board):
    """Print an animated game-over report for terminal visibility."""
    outcome = chess_board.outcome(claim_draw=True)
    termination = outcome.termination.name.replace("_", " ").title() if outcome else "Unknown"
    result = outcome.result() if outcome else chess_board.result(claim_draw=True)

    if result == "1-0":
        winner_line = "Winner: White"
    elif result == "0-1":
        winner_line = "Winner: Black"
    else:
        winner_line = "Winner: None (Draw)"

    if chess_board.is_checkmate():
        reason_line = "Reason: Checkmate delivered. The side to move has no legal escape."
    elif chess_board.is_stalemate():
        reason_line = "Reason: Stalemate. No legal moves remain, but the king is not in check."
    elif chess_board.is_insufficient_material():
        reason_line = "Reason: Draw by insufficient material."
    elif chess_board.is_fifty_moves():
        reason_line = "Reason: Draw by fifty-move rule."
    elif chess_board.is_repetition(3):
        reason_line = "Reason: Draw by threefold repetition."
    else:
        reason_line = f"Reason: {termination}."

    def _type_line(text: str, delay_s: float = 0.01) -> None:
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(delay_s)
        sys.stdout.write("\n")
        sys.stdout.flush()

    frames = [
        ("#" * 72, "GAME OVER".center(72), "#" * 72),
        ("*" * 72, "CHECKMATE CHRONICLE".center(72), "*" * 72),
        ("=" * 72, "FINAL POSITION LOCKED".center(72), "=" * 72),
    ]

    print()
    for top, middle, bottom in frames:
        print(top)
        print(middle)
        print(bottom)
        time.sleep(0.12)

    _type_line(f"Final result      : {result}", delay_s=0.008)
    _type_line(f"{winner_line}", delay_s=0.008)
    _type_line(f"{reason_line}", delay_s=0.006)
    _type_line(f"Termination type  : {termination}", delay_s=0.008)
    _type_line(f"Final FEN         : {chess_board.fen()}", delay_s=0.003)
    print("=" * 72 + "\n")

def save_game_state(chess_board):
    """Save the current board FEN to stored_game.txt."""
    if chess_board is not None:
        with open(SAVED_GAME_FILE, "w") as f:
            f.write(chess_board.fen())
        print(f"\nGame saved to {SAVED_GAME_FILE}")


def load_saved_game():
    """Return the FEN string from stored_game.txt, or None if it doesn't exist."""
    if os.path.exists(SAVED_GAME_FILE):
        with open(SAVED_GAME_FILE, "r") as f:
            return f.read().strip()
    return None


def show_board_and_wait(chess_board):
    """Render the board as SVG in browser, wait for Enter, then clean up the file."""
    visualize_board(chess_board)
    input("Board displayed in browser. Press Enter to start the game loop...")
    if os.path.exists("current_board.svg"):
        os.remove("current_board.svg")


def prompt_continue_saved_game():
    """
    If a saved game file exists, ask the user if they want to continue it.
    Returns (chess_board, resumed) where resumed=True means we loaded the save.
    """
    saved_fen = load_saved_game()
    if saved_fen:
        print(f"\nSaved game found in '{SAVED_GAME_FILE}'.")
        print(f"  FEN: {saved_fen}")
        answer = input("Continue saved game? [y/N]: ").strip().lower()
        if answer == "y":
            board = chess.Board(saved_fen)
            print("Resuming saved game.")
            show_board_and_wait(board)
            return board, True
        else:
            os.remove(SAVED_GAME_FILE)
            print("Saved game deleted. Starting fresh.")
    return None, False


# Global reference so the signal handler can access the board
_chess_board_ref = [None]


def _exit_handler(signum, frame):
    save_game_state(_chess_board_ref[0])
    sys.exit(0)


# Register save-on-exit for SIGINT (Ctrl+C) and SIGTERM
signal.signal(signal.SIGINT, _exit_handler)
signal.signal(signal.SIGTERM, _exit_handler)


def detect_castling(chess_board, removals):
    """
    Given two removal squares (row, col) for one color, check if any legal castle
    move matches. Returns the UCI string (e.g. 'e1g1') or None.
    """
    removed_squares = {row_col_to_chess_square(r, c) for r, c in [tuple(removals[0]), tuple(removals[1])]}
    for move in chess_board.legal_moves:
        if chess_board.is_castling(move):
            # Castling UCI: king from/to — the rook squares are implied
            king_from = move.from_square
            king_to = move.to_square
            # The rook's from-square depends on kingside vs queenside
            if chess_board.is_kingside_castling(move):
                rook_from = chess.H1 if chess_board.turn == chess.WHITE else chess.H8
            else:
                rook_from = chess.A1 if chess_board.turn == chess.WHITE else chess.A8
            if removed_squares == {king_from, rook_from}:
                return move.uci()
    return None


def main():
    faulthandler.enable(all_threads=True)
    zed = ZedCamera()
    detector = Detector(families='tag36h11 tag25h9')
    camera_intrinsic = zed.camera_intrinsic

    prior_board_state = None

    # --- Saved game check ---
    chess_board, resumed = prompt_continue_saved_game()
    turn = chess_board.turn if resumed else chess.WHITE
    _chess_board_ref[0] = chess_board  # keep signal handler in sync

    try:
        for i in range(40):
            print(f"\n--- Iteration {i + 1}/40 ---")
            loop_start = time.time()

            cv_image = zed.image
            board_state, warped_with_pieces, resized_raw = get_board_state(cv_image, detector, camera_intrinsic)

            if board_state is None:
                print("Could not detect board this iteration, skipping.")
            else:
                # Initialize chess board from first good detection (skip if resumed from save)
                if chess_board is None:
                    fen = detect_starting_fen(board_state)
                    if fen is None:
                        print("WARNING: Unknown board state. Set CURRENT_FEN at the top of game_loop.py and restart.")
                        break
                    chess_board = chess.Board(fen)
                    _chess_board_ref[0] = chess_board
                    print(f"Chess board initialized from FEN: {fen}")

                if prior_board_state is not None:
                    one_removals, two_removals, one_additions, two_additions = compare_board_states(prior_board_state, board_state)
                    changed = any(len(x) > 0 for x in [one_removals, two_removals, one_additions, two_additions])

                    if changed:
                        # Check whose pieces moved: color_val 1=black, 2=white
                        moving_color_val = None
                        if (len(one_removals) > 0 or len(one_additions) > 0) and len(two_removals) == 0 and len(two_additions) == 0:
                            moving_color_val = 1  # black moved
                        elif (len(two_removals) > 0 or len(two_additions) > 0) and len(one_removals) == 0 and len(one_additions) == 0:
                            moving_color_val = 2  # white moved

                        # Wrong turn check
                        expected_color_val = 2 if turn == chess.WHITE else 1
                        if moving_color_val is not None and moving_color_val != expected_color_val:
                            wrong_color = "black" if moving_color_val == 1 else "white"
                            right_color = "white" if turn == chess.WHITE else "black"
                            print(f"Wrong turn: {wrong_color} moved but it is {right_color}'s turn.")
                            all_removals = [(tuple(rc), 1) for rc in one_removals] + [(tuple(rc), 2) for rc in two_removals]
                            for rc, color_val in all_removals:
                                sq = row_col_to_chess_square(*rc)
                                piece = chess_board.piece_at(sq)
                                color_name = "white" if color_val == 2 else "black"
                                piece_name = chess.piece_name(piece.piece_type) if piece else "unknown"
                                print(f"  return {color_name} {piece_name} to {chess.square_name(sq)}")
                        else:
                            # Determine if this is a valid single move or legal capture
                            is_single_move = (
                                (len(one_removals) == 1 and len(one_additions) == 1 and len(two_removals) == 0 and len(two_additions) == 0) or
                                (len(two_removals) == 1 and len(two_additions) == 1 and len(one_removals) == 0 and len(one_additions) == 0)
                            )
                            is_legal_capture = (
                                (len(one_removals) == 1 and len(one_additions) == 1 and len(two_removals) == 1 and len(two_additions) == 0) or
                                (len(two_removals) == 1 and len(two_additions) == 1 and len(one_removals) == 1 and len(one_additions) == 0)
                            )
                            is_castle = (
                                (len(one_removals) == 2 and len(one_additions) == 2 and len(two_removals) == 0 and len(two_additions) == 0) or
                                (len(two_removals) == 2 and len(two_additions) == 2 and len(one_removals) == 0 and len(one_additions) == 0)
                            )

                            if is_castle:
                                castle_color_val = 1 if len(one_removals) == 2 else 2
                                removals = one_removals if castle_color_val == 1 else two_removals
                                castle_move = detect_castling(chess_board, removals)
                                if castle_move is None:
                                    print("Looks like castling but no legal castle move found. Please return pieces.")
                                else:
                                    print(f"Castling detected! UCI move: {castle_move}")
                                    try:
                                        chess_board.push_uci(castle_move)
                                        _chess_board_ref[0] = chess_board
                                        turn = chess.BLACK if turn == chess.WHITE else chess.WHITE
                                        prior_board_state = board_state
                                        print(f"Castling applied: {castle_move}")
                                    except Exception as e:
                                        print(f"  ILLEGAL CASTLE ({e}): please return pieces to original squares.")
                                        continue

                                    # Robot responds if it was white who castled
                                    if turn == chess.BLACK:
                                        try:
                                            robot_move = get_best_move(chess_board.fen(), time_limit=2.0)
                                            move_string = robot_move.uci()
                                            print(f"Sending move {move_string} to robot arm...")
                                            from_square, to_square, from_occupant, to_occupant = parse_move_string(chess_board, move_string)
                                            if to_occupant is not None:
                                                capture_piece(from_occupant, to_occupant, from_square, to_square, zed, capture_count)
                                                capture_count += 1
                                            else:
                                                move_piece(from_occupant, from_square, to_square, zed)
                                            chess_board.push_uci(move_string)
                                            _chess_board_ref[0] = chess_board
                                            turn = chess.WHITE
                                            print(f"Robot played {move_string}. White's turn.")
                                        except Exception as e:
                                            print(f"  Robot move failed: {e}")

                            elif not is_single_move and not is_legal_capture:
                                all_removals = [(tuple(rc), 1) for rc in one_removals] + [(tuple(rc), 2) for rc in two_removals]
                                if all_removals:
                                    print("Invalid board change. Please return pieces to their original squares:")
                                    for rc, color_val in all_removals:
                                        sq = row_col_to_chess_square(*rc)
                                        piece = chess_board.piece_at(sq)
                                        color_name = "white" if color_val == 2 else "black"
                                        piece_name = chess.piece_name(piece.piece_type) if piece else "unknown"
                                        print(f"  return {color_name} {piece_name} to {chess.square_name(sq)}")
                            else:
                                move = chess_utils.determine_move(one_removals, two_removals, one_additions, two_additions)
                                print(f"Change detected! UCI move: {move}")
                                describe_move(chess_board, (one_removals, two_removals), (one_additions, two_additions))

                                try:
                                    chess_board.push_uci(move)
                                    _chess_board_ref[0] = chess_board
                                    turn = chess.BLACK if turn == chess.WHITE else chess.WHITE
                                    if chess_board.is_check():
                                        checked_side = "White" if chess_board.turn == chess.WHITE else "Black"
                                        print(f"Check: {checked_side} is in check.")
                                    prior_board_state = board_state
                                    if chess_board.is_game_over(claim_draw=True):
                                        print_game_over_banner(chess_board)
                                        break
                                except Exception as e:
                                    print(f"  ILLEGAL MOVE ({e}): please return pieces to original squares.")
                                    all_removals = [(tuple(rc), 1) for rc in one_removals] + [(tuple(rc), 2) for rc in two_removals]
                                    for rc, color_val in all_removals:
                                        sq = row_col_to_chess_square(*rc)
                                        piece = chess_board.piece_at(sq)
                                        color_name = "white" if color_val == 2 else "black"
                                        piece_name = chess.piece_name(piece.piece_type) if piece else "unknown"
                                        print(f"  return {color_name} {piece_name} to {chess.square_name(sq)}")
                                    

                                # After a successful white move, it is now black's turn — robot moves
                                if turn == chess.BLACK:
                                    try:
                                        robot_move = get_best_move(chess_board.fen(), time_limit=2.0)
                                        move_string = robot_move.uci()
                                        print(f"Sending move {move_string} to robot arm...")

                                        from_square, to_square, from_occupant, to_occupant = parse_move_string(chess_board, move_string)

                                        if to_occupant is not None:
                                            capture_count = count_white_captures(chess_board)
                                            capture_piece(from_occupant, to_occupant, from_square, to_square, zed, capture_count)
                                        else:
                                            move_piece(from_occupant, from_square, to_square, zed)

                                        # Commit robot's move to the board and flip turn back to white
                                        chess_board.push_uci(move_string)
                                        _chess_board_ref[0] = chess_board
                                        turn = chess.WHITE
                                        if chess_board.is_check():
                                            checked_side = "White" if chess_board.turn == chess.WHITE else "Black"
                                            print(f"Check: {checked_side} is in check.")
                                        if chess_board.is_game_over(claim_draw=True):
                                            print_game_over_banner(chess_board)
                                            break
                                        # Refresh baseline from a post-robot frame so the next loop
                                        # does not re-detect the robot's own move as a new change.
                                        # Keep trying until we successfully get a board state.
                                        post_robot_state = None
                                        while post_robot_state is None:
                                            post_robot_image = zed.image
                                            post_robot_state, _, _ = get_board_state(post_robot_image, detector, camera_intrinsic)
                                            if post_robot_state is None:
                                                print("Post-robot board capture failed; retrying in 0.5s...")
                                                time.sleep(0.5)
                                        prior_board_state = post_robot_state
                                        print(f"Robot played {move_string}. White's turn.")
                                    except Exception as e:
                                        print(f"  Robot move failed: {e}")
                    else:
                        print("No change detected.")
                else:
                    prior_board_state = board_state
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
