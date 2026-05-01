"""Standalone debug runner for visual board-state detection."""

from __future__ import annotations

import cv2
from pupil_apriltags import Detector

import chess_utils
from piece_continuity import compare_board_states, display_board_state, get_board_state
from utils.zed_camera import ZedCamera


def main() -> None:
    """Continuously capture and print predicted move deltas until 'k' is pressed."""
    zed = ZedCamera()
    detector = Detector(families="tag36h11 tag25h9")
    camera_intrinsic = zed.camera_intrinsic
    prior_board_state = None

    try:
        while True:
            cv_image = zed.image
            board_state, warped_with_pieces, resized_raw = get_board_state(cv_image, detector, camera_intrinsic)

            if board_state is None:
                print("Board not detected, retrying...")
                if resized_raw is not None:
                    cv2.imshow("Robot Calibration", resized_raw)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                continue

            display_board_state(warped_with_pieces, resized_raw)
            cv2.destroyAllWindows()

            if prior_board_state is not None:
                one_removals, two_removals, one_additions, two_additions = compare_board_states(prior_board_state, board_state)
                predicted_move = chess_utils.determine_move(one_removals, two_removals, one_additions, two_additions)
                print(f"one_removals: {one_removals}")
                print(f"two_removals: {two_removals}")
                print(f"one_additions: {one_additions}")
                print(f"two_additions: {two_additions}")
                print(f"predicted move: {predicted_move}")

            prior_board_state = board_state
            if cv2.waitKey(1) == ord("k"):
                break
    finally:
        cv2.destroyAllWindows()
        zed.close()


if __name__ == "__main__":
    main()
