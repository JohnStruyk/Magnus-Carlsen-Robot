import cv2
import numpy as np
from pupil_apriltags import Detector

from utils.zed_camera import ZedCamera
from piece_continuity import get_board_state, display_board_state, compare_board_states
import chess_utils

CAMERA_INTRINSIC = np.array(((1062.18, 0, 1047.36), (0, 1062.18, 610.32), (0, 0, 1)))


def main():
    zed = ZedCamera()
    detector = Detector(families='tag36h11 tag25h9')

    prior_board_state = None

    for i in range(10):
        print(f"\n--- Iteration {i + 1}/10 ---")

        cv_image = zed.image
        board_state, warped_with_pieces, resized_raw = get_board_state(cv_image, detector, CAMERA_INTRINSIC)

        if board_state is None:
            print("Could not detect board this iteration, skipping.")
            continue

        display_board_state(warped_with_pieces, resized_raw)
        cv2.destroyAllWindows()

        if prior_board_state is not None:
            one_removals, two_removals, one_additions, two_additions = compare_board_states(
                prior_board_state, board_state
            )
            changed = any(len(x) > 0 for x in [one_removals, two_removals, one_additions, two_additions])
            if changed:
                move = chess_utils.determine_move(one_removals, two_removals, one_additions, two_additions)
                print(f"Change detected! Predicted move: {move}")
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
