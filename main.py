"""hand-sketch: webcam app for hand-tracked drawing.

Phase 3: drawing is live. DRAW mode (one finger up, thumb away) leaves
cyan trails on a black canvas. PINCH and PALM are classified by the HUD
but do not change anything yet — those become GRAB and VIEW in later phases.

Run:  python main.py
Quit: press 'q' in the window.
"""

import cv2

from document import Document
from gestures import GestureStateMachine, classify_raw
from modes import DrawHandler, EraseHandler, GrabHandler, ViewHandler
from render import render_frame
from tracker import HandTracker
from viewport import Viewport


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    tracker = HandTracker()
    state_machine = GestureStateMachine(debounce_frames=4)
    document = Document()
    viewport = Viewport()

    draw_handler = DrawHandler(smoothing_alpha=0.5)
    erase_handler = EraseHandler()
    grab_handler = GrabHandler()
    view_handler = ViewHandler()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror so on-screen hand moves the same way as your physical hand.
            frame = cv2.flip(frame, 1)

            landmarks = tracker.process(frame)
            raw_gesture, features = classify_raw(landmarks)
            mode, transitions = state_machine.update(raw_gesture)

            draw_handler.handle(mode, raw_gesture, transitions, landmarks, document, viewport)
            erase_handler.handle(mode, raw_gesture, transitions, landmarks, document, viewport)
            grab_handler.handle(mode, raw_gesture, transitions, landmarks, document, viewport)
            view_handler.handle(mode, raw_gesture, transitions, landmarks, document, viewport)

            output = render_frame(
                frame, document, viewport, landmarks, mode, raw_gesture, features
            )
            cv2.imshow("hand-sketch (q to quit)", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()
