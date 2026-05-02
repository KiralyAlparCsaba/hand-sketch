"""hand-sketch: webcam app for hand-tracked drawing.

Phase 1 (this file): verify the MediaPipe + webcam pipeline by drawing the
hand skeleton on the live feed. No gestures, canvas, or glow yet.

Run:  python main.py
Quit: press 'q' in the window.
"""

import cv2

from tracker import HAND_CONNECTIONS, HandTracker


# OpenCV uses BGR, not RGB.
COLOR_LANDMARK = (0, 255, 255)    # yellow dots
COLOR_CONNECTION = (255, 255, 0)  # cyan lines


def draw_hand(frame, landmarks):
    """Overlay the hand skeleton on `frame` in-place."""
    pts = landmarks.pixels
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), COLOR_CONNECTION, 2)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 4, COLOR_LANDMARK, -1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    tracker = HandTracker()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror so on-screen hand moves the same way as your physical hand.
            frame = cv2.flip(frame, 1)

            landmarks = tracker.process(frame)
            if landmarks is not None:
                draw_hand(frame, landmarks)
                label = f"{landmarks.handedness} hand"
                color = (255, 255, 255)
            else:
                label = "no hand"
                color = (128, 128, 128)

            cv2.putText(
                frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

            cv2.imshow("hand-sketch (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()
