"""hand-sketch: webcam app for hand-tracked drawing.

Phase 2 (this file): hand detection + gesture classification with HUD.
No canvas or glow yet — just verify the gesture pipeline by watching the
mode label change as you make DRAW / PINCH / PALM poses.

Run:  python main.py
Quit: press 'q' in the window.
"""

import cv2

from tracker import HAND_CONNECTIONS, HandTracker
from gestures import Gesture, GestureStateMachine, classify_raw


# OpenCV uses BGR, not RGB.
COLOR_LANDMARK = (0, 255, 255)    # yellow dots
COLOR_CONNECTION = (255, 255, 0)  # cyan lines

# Mode -> BGR color for the HUD label.
MODE_COLORS = {
    Gesture.DRAW:  (255, 255,   0),  # cyan
    Gesture.PINCH: (  0, 255, 255),  # yellow
    Gesture.PALM:  (255,   0, 255),  # magenta
    Gesture.NONE:  (160, 160, 160),  # gray
}


def draw_hand(frame, landmarks):
    """Overlay the hand skeleton on `frame` in-place."""
    pts = landmarks.pixels
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), COLOR_CONNECTION, 2)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 4, COLOR_LANDMARK, -1)


def draw_hud(frame, mode, raw, features, handedness):
    """Top-left status block: current debounced mode, raw classification,
    pinch distance, finger-extension flags, and handedness."""
    # Big mode label.
    cv2.putText(
        frame, f"MODE: {mode.name}", (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, MODE_COLORS[mode], 2,
    )

    # Raw classification underneath, dimmer.
    raw_dim = tuple(c // 2 for c in MODE_COLORS[raw])
    cv2.putText(
        frame, f"raw: {raw.name}", (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, raw_dim, 1,
    )

    # Diagnostics (only when a hand is visible).
    if features is not None:
        ext_str = "".join([
            "I" if features.index_extended  else ".",
            "M" if features.middle_extended else ".",
            "R" if features.ring_extended   else ".",
            "P" if features.pinky_extended  else ".",
        ])
        cv2.putText(
            frame,
            f"pinch: {features.pinch_distance:.2f}  fingers: {ext_str}",
            (10, 82),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

    cv2.putText(
        frame, handedness, (10, 104),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    tracker = HandTracker()
    state_machine = GestureStateMachine(debounce_frames=4)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror so on-screen hand moves the same way as your physical hand.
            frame = cv2.flip(frame, 1)

            landmarks = tracker.process(frame)
            raw_gesture, features = classify_raw(landmarks)
            mode, _transitions = state_machine.update(raw_gesture)

            handedness = (
                f"{landmarks.handedness} hand" if landmarks is not None else "no hand"
            )

            if landmarks is not None:
                draw_hand(frame, landmarks)

            draw_hud(frame, mode, raw_gesture, features, handedness)

            cv2.imshow("hand-sketch (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()
