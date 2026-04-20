import cv2
import numpy as np
import chess
from pupil_apriltags import Detector

from utils.zed_camera import ZedCamera
from piece_continuity import get_board_state, display_board_state, compare_board_states
import chess_utils

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
    if np.array_equal(board_state, STANDARD_BOARD_STATE):
        print("Standard starting position detected.")
        return STANDARD_STARTING_FEN

    if CURRENT_FEN:
        print("Using CURRENT_FEN.")
        return CURRENT_FEN

    return None


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


def main():
    zed = ZedCamera()
    detector = Detector(families='tag36h11 tag25h9')
    camera_intrinsic = zed.camera_intrinsic

    prior_board_state = None
    chess_board = None
    missing_pieces = {}  # chess.Square -> consecutive cycles missing

    for i in range(10):
        print(f"\n--- Iteration {i + 1}/10 ---")

        cv_image = zed.image
        board_state, warped_with_pieces, resized_raw = get_board_state(cv_image, detector, camera_intrinsic)

        if board_state is None:
            print("Could not detect board this iteration, skipping.")
            continue

        display_board_state(warped_with_pieces, resized_raw)
        cv2.destroyAllWindows()

        # Initialize chess board from first good detection
        if chess_board is None:
            fen = detect_starting_fen(board_state)
            if fen is None:
                print("WARNING: Unknown board state. Set CURRENT_FEN at the top of game_loop.py and restart.")
                break
            chess_board = chess.Board(fen)
            print(f"Chess board initialized from FEN: {fen}")

        if prior_board_state is not None:
            # --- Piece presence check ---
            abort = False
            for sq in chess.SQUARES:
                piece = chess_board.piece_at(sq)
                if piece is None:
                    continue
                row = 7 - chess.square_rank(sq)
                col = chess.square_file(sq)
                expected_color_val = 2 if piece.color == chess.WHITE else 1
                if board_state[row, col] != expected_color_val:
                    missing_pieces[sq] = missing_pieces.get(sq, 0) + 1
                    if missing_pieces[sq] >= 2:
                        color_name = "white" if piece.color == chess.WHITE else "black"
                        piece_name = chess.piece_name(piece.piece_type)
                        sq_name = chess.square_name(sq)
                        print(f"ABORT: {color_name} {piece_name} on {sq_name} has been missing for 2 cycles.")
                        abort = True
                else:
                    missing_pieces.pop(sq, None)

            if abort:
                break

            one_removals, two_removals, one_additions, two_additions = compare_board_states(
                prior_board_state, board_state
            )
            changed = any(len(x) > 0 for x in [one_removals, two_removals, one_additions, two_additions])
            if changed:
                move = chess_utils.determine_move(one_removals, two_removals, one_additions, two_additions)
                print(f"Change detected! UCI move: {move}")
                describe_move(chess_board, (one_removals, two_removals), (one_additions, two_additions))

                # Advance the chess board if the move is valid UCI
                try:
                    chess_board.push_uci(move)
                except Exception:
                    print(f"  ILLEGAL MOVE: (Could not apply move '{move}' to chess board)")
                    # TODO: MOVE ILLEGAL PIECE BACK

            else:
                print("No change detected.")
        else:
            print("No prior state to compare against.")

        prior_board_state = board_state

    zed.close()
    cv2.destroyAllWindows()
    print("\nGame loop finished.")


if __name__ == "__main__":
    main()
