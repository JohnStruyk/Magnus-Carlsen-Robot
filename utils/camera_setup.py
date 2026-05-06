"""Sanity-check both AprilTag families before trusting vision in anger."""

import cv2

from utils.zed_camera import ZedCamera
from utils.calibrate_tags import (
    detect_playmat_and_chessboard_tags,
    best_tag_per_id_0_3,
    draw_dual_family_tag_overlays,
    to_bgr_display,
    resize_for_preview,
    PLAYMAT_TAG_FAMILY,
    CHESSBOARD_TAG_FAMILY,
)


def overlay_status_lines(pm_ok: bool, ch_ok: bool, playmat_raw, chess_raw):
    pm_ids = sorted(set(int(t.tag_id) for t in playmat_raw))  # ids seen this frame on playmat sheet
    ch_ids = sorted(set(int(t.tag_id) for t in chess_raw))  # ids seen on chessboard sheet
    pm_status = "OK (4/4)" if pm_ok else f"MISSING ({len(pm_ids)}/4) ids={pm_ids}"
    ch_status = "OK (4/4)" if ch_ok else f"MISSING ({len(ch_ids)}/4) ids={ch_ids}"
    return [
        f"Playmat  [{PLAYMAT_TAG_FAMILY}]: {pm_status}",
        f"Chessboard [{CHESSBOARD_TAG_FAMILY}]: {ch_status}",
        "Press any key to quit",
    ]


def run_preview_loop(zed: ZedCamera) -> None:
    print("Camera running. Press any key in the window to quit.")
    print(f"  Playmat family  : {PLAYMAT_TAG_FAMILY}  (shown in green)")
    print(f"  Chessboard family: {CHESSBOARD_TAG_FAMILY}  (shown in orange)")
    print()

    try:
        while True:
            img = zed.image
            if img is None:
                continue

            # Gray image ignored here (overlay only needs tag lists).
            _, playmat_raw, chess_raw = detect_playmat_and_chessboard_tags(img)
            _, pm_ok = best_tag_per_id_0_3(playmat_raw)
            _, ch_ok = best_tag_per_id_0_3(chess_raw)

            bgr = to_bgr_display(img).copy()
            draw_dual_family_tag_overlays(bgr, playmat_raw, chess_raw)

            for i, line in enumerate(overlay_status_lines(pm_ok, ch_ok, playmat_raw, chess_raw)):
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


def main() -> None:
    zed = ZedCamera()
    run_preview_loop(zed)


if __name__ == "__main__":
    main()
