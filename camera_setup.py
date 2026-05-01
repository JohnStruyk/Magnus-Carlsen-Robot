"""
Camera setup check — shows a live annotated view of what the camera sees.
Green tags = playmat family, orange tags = chessboard family.
Prints whether all 4 corners (ids 0-3) are visible for each family.
Press any key to quit.
"""

import cv2

from utils.zed_camera import ZedCamera
from calibrate_tags import (
    detect_playmat_and_chessboard_tags,
    best_tag_per_id_0_3,
    draw_dual_family_tag_overlays,
    to_bgr_display,
    resize_for_preview,
    PLAYMAT_TAG_FAMILY,
    CHESSBOARD_TAG_FAMILY,
)


def main() -> None:
    """Run a live camera diagnostics view for dual-family AprilTag visibility."""
    zed = ZedCamera()
    print("Camera running. Press any key in the window to quit.")
    print(f"  Playmat family  : {PLAYMAT_TAG_FAMILY}  (shown in green)")
    print(f"  Chessboard family: {CHESSBOARD_TAG_FAMILY}  (shown in orange)")
    print()

    try:
        while True:
            img = zed.image
            if img is None:
                continue

            _, playmat_raw, chess_raw = detect_playmat_and_chessboard_tags(img)
            _, pm_ok = best_tag_per_id_0_3(playmat_raw)
            _, ch_ok = best_tag_per_id_0_3(chess_raw)

            bgr = to_bgr_display(img).copy()
            draw_dual_family_tag_overlays(bgr, playmat_raw, chess_raw)

            pm_ids = sorted(set(int(t.tag_id) for t in playmat_raw))
            ch_ids = sorted(set(int(t.tag_id) for t in chess_raw))
            pm_status = "OK (4/4)" if pm_ok else f"MISSING ({len(pm_ids)}/4) ids={pm_ids}"
            ch_status = "OK (4/4)" if ch_ok else f"MISSING ({len(ch_ids)}/4) ids={ch_ids}"

            # Overlay status text onto the image
            lines = [
                f"Playmat  [{PLAYMAT_TAG_FAMILY}]: {pm_status}",
                f"Chessboard [{CHESSBOARD_TAG_FAMILY}]: {ch_status}",
                "Press any key to quit",
            ]
            for i, line in enumerate(lines):
                y = 36 + i * 32
                cv2.putText(bgr, line, (12, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                color = (0, 220, 0) if "OK" in line else (0, 60, 220)
                cv2.putText(bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            cv2.imshow("Camera Setup", resize_for_preview(bgr))
            if cv2.waitKey(1) != -1:
                break
    finally:
        cv2.destroyAllWindows()
        zed.close()


if __name__ == "__main__":
    main()
