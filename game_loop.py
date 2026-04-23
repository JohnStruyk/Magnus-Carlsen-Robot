import cv2
import time
import numpy as np
import chess
from pupil_apriltags import Detector

from utils.zed_camera import ZedCamera
from piece_continuity import get_board_state, display_board_state, compare_board_states
import chess_utils
from stockfish_int import get_best_move
from pickup_board_piece import move_piece, capture_piece

CAPTURE_INTERVAL = 5  # seconds between captures

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

def main():
    zed = ZedCamera()
    detector = Detector(families='tag36h11 tag25h9')
    camera_intrinsic = zed.camera_intrinsic

    prior_board_state = None
    chess_board = None
    turn = chess.WHITE  # white moves first

    capture_count = 0

    for i in range(40):
        print(f"\n--- Iteration {i + 1}/40 ---")
        loop_start = time.time()

        cv_image = zed.image
        board_state, warped_with_pieces, resized_raw = get_board_state(cv_image, detector, camera_intrinsic)

        if board_state is None:
            print("Could not detect board this iteration, skipping.")
        else:
            # Initialize chess board from first good detection
            if chess_board is None:
                fen = detect_starting_fen(board_state)
                if fen is None:
                    print("WARNING: Unknown board state. Set CURRENT_FEN at the top of game_loop.py and restart.")
                    break
                chess_board = chess.Board(fen)
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
                        # display_board_state(warped_with_pieces, resized_raw)
                        # cv2.destroyAllWindows()
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

                        if not is_single_move and not is_legal_capture:
                            all_removals = [(tuple(rc), 1) for rc in one_removals] + [(tuple(rc), 2) for rc in two_removals]
                            if all_removals:
                                print("Invalid board change. Please return pieces to their original squares:")
                                for rc, color_val in all_removals:
                                    sq = row_col_to_chess_square(*rc)
                                    piece = chess_board.piece_at(sq)
                                    color_name = "white" if color_val == 2 else "black"
                                    piece_name = chess.piece_name(piece.piece_type) if piece else "unknown"
                                    print(f"  return {color_name} {piece_name} to {chess.square_name(sq)}")
                            # display_board_state(warped_with_pieces, resized_raw)
                            # cv2.destroyAllWindows()
                        else:
                            move = chess_utils.determine_move(one_removals, two_removals, one_additions, two_additions)
                            print(f"Change detected! UCI move: {move}")
                            describe_move(chess_board, (one_removals, two_removals), (one_additions, two_additions))

                            try:
                                chess_board.push_uci(move)
                                turn = chess.BLACK if turn == chess.WHITE else chess.WHITE
                                prior_board_state = board_state

                                # After a successful white move, it is now black's turn
                                if turn == chess.BLACK:
                                    # TODO: Insert robot manipulation logic here.
                                    # The robot should calculate and execute black's move physically on the board.
                                    robot_move = get_best_move(chess_board.fen(), time_limit=2.0)

                                    move_string = robot_move.uci() 
                                    print(f"Sending move {move_string} to robot arm...")

                                    # TODO: Map move string to robot commands,update chess_board once move is executed

                                    from_square, to_square, from_occupant, to_occupant = parse_move_string(chess_board, move_string)
                                    
                                    if to_occupant is not None:
                                        capture_piece(from_occupant, to_occupant, from_square, to_square, zed, capture_count)
                                        capture_count += 1
                                    else:
                                        move_piece(from_occupant, from_square, to_square, zed)

                                    # For now, display the board and wait for a human to move black's piece manually.
                                    print("Black's turn. Move black's piece, then press any key to continue.")
                                    display_board_state(warped_with_pieces, resized_raw)
                                    cv2.destroyAllWindows()
                            except Exception:
                                all_removals = [(tuple(rc), 1) for rc in one_removals] + [(tuple(rc), 2) for rc in two_removals]
                                for rc, color_val in all_removals:
                                    sq = row_col_to_chess_square(*rc)
                                    piece = chess_board.piece_at(sq)
                                    color_name = "white" if color_val == 2 else "black"
                                    piece_name = chess.piece_name(piece.piece_type) if piece else "unknown"
                                    print(f"  ILLEGAL MOVE: return {color_name} {piece_name} to {chess.square_name(sq)}")
                            # display_board_state(warped_with_pieces, resized_raw)
                            # cv2.destroyAllWindows()
                else:
                    print("No change detected.")
            else:
                prior_board_state = board_state
                print("No prior state to compare against.")

        # display_board_state(warped_with_pieces, resized_raw)
        # cv2.destroyAllWindows()

        elapsed = time.time() - loop_start
        sleep_time = max(0, CAPTURE_INTERVAL - elapsed)
        print(f"Sleeping {sleep_time:.1f}s until next capture.")
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()
