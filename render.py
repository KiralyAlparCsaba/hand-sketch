"""Frame rendering: black canvas + strokes + hand overlay + HUD.

Phase 3: plain cyan polylines (no glow yet). Glow comes in a later phase
as a Gaussian-blur pass over the strokes layer before compositing.

The webcam frame is NOT drawn — only its dimensions are used to size the
output canvas. The artistic design is a black background with only the
hand outline and strokes visible.
"""

from typing import Optional

import cv2
import numpy as np

from document import Document
from gestures import Gesture, GestureFeatures, INDEX_TIP
from modes import ERASER_RADIUS
from tracker import HAND_CONNECTIONS, HandLandmarks
from viewport import Viewport


# OpenCV uses BGR (not RGB).
COLOR_STROKE = (255, 255, 0)        # cyan
COLOR_LANDMARK = (0, 255, 255)      # yellow joint dots
COLOR_CONNECTION = (255, 200, 0)    # cyan-ish skeleton lines
COLOR_HUD_DIM = (200, 200, 200)

STROKE_THICKNESS = 3

MODE_COLORS = {
    Gesture.DRAW:  (255, 255,   0),  # cyan
    Gesture.PINCH: (  0, 255, 255),  # yellow
    Gesture.PALM:  (255,   0, 255),  # magenta
    Gesture.ERASE: (  0,   0, 255),  # red
    Gesture.NONE:  (160, 160, 160),  # gray
}

COLOR_ERASER_OUTLINE = (0, 0, 255)   # red ring around eraser
COLOR_ERASER_FILL = (0, 0, 200)      # darker red translucent fill


def render_frame(
    frame: np.ndarray,
    document: Document,
    viewport: Viewport,
    landmarks: Optional[HandLandmarks],
    mode: Gesture,
    raw: Gesture,
    features: Optional[GestureFeatures],
) -> np.ndarray:
    """Composite the output image. Starts from a black canvas; layers in
    strokes, hand overlay, and HUD. Returns a new image — does not modify
    the input frame."""
    h, w = frame.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    _draw_strokes(canvas, document, viewport)
    if landmarks is not None:
        _draw_hand(canvas, landmarks)
        if mode == Gesture.ERASE:
            _draw_eraser_cursor(canvas, landmarks, viewport)
    _draw_hud(canvas, mode, raw, features, landmarks)
    return canvas


def _draw_eraser_cursor(
    canvas: np.ndarray, landmarks: HandLandmarks, viewport: Viewport
) -> None:
    """Translucent red disc + outline ring at the index fingertip,
    sized to match the actual erase radius in canvas units."""
    tip = landmarks.pixels[INDEX_TIP]
    radius_screen = max(1, int(round(ERASER_RADIUS * viewport.zoom)))
    center = (int(tip[0]), int(tip[1]))

    # Translucent fill: blend an overlay so we don't fully obscure strokes.
    overlay = canvas.copy()
    cv2.circle(overlay, center, radius_screen, COLOR_ERASER_FILL, thickness=-1)
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    # Crisp outline.
    cv2.circle(canvas, center, radius_screen, COLOR_ERASER_OUTLINE, thickness=2)


def _draw_strokes(
    canvas: np.ndarray, document: Document, viewport: Viewport
) -> None:
    for group in document.all_groups():
        for stroke in group.strokes:
            if len(stroke) < 2:
                continue
            pts_canvas = np.array(stroke.points, dtype=np.float32)
            pts_screen = viewport.canvas_polyline_to_screen(pts_canvas)
            pts_int = pts_screen.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                canvas, [pts_int], isClosed=False,
                color=COLOR_STROKE, thickness=STROKE_THICKNESS,
                lineType=cv2.LINE_AA,
            )


def _draw_hand(canvas: np.ndarray, landmarks: HandLandmarks) -> None:
    pts = landmarks.pixels
    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), COLOR_CONNECTION, 2)
    for x, y in pts:
        cv2.circle(canvas, (int(x), int(y)), 4, COLOR_LANDMARK, -1)


def _draw_hud(
    canvas: np.ndarray,
    mode: Gesture,
    raw: Gesture,
    features: Optional[GestureFeatures],
    landmarks: Optional[HandLandmarks],
) -> None:
    cv2.putText(
        canvas, f"MODE: {mode.name}", (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, MODE_COLORS[mode], 2,
    )
    raw_dim = tuple(c // 2 for c in MODE_COLORS[raw])
    cv2.putText(
        canvas, f"raw: {raw.name}", (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, raw_dim, 1,
    )
    if features is not None:
        ext_str = "".join([
            "I" if features.index_extended  else ".",
            "M" if features.middle_extended else ".",
            "R" if features.ring_extended   else ".",
            "P" if features.pinky_extended  else ".",
        ])
        cv2.putText(
            canvas,
            f"pinch: {features.pinch_distance:.2f}  fingers: {ext_str}",
            (10, 82),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HUD_DIM, 1,
        )
    label = (
        f"{landmarks.handedness} hand" if landmarks is not None else "no hand"
    )
    cv2.putText(
        canvas, label, (10, 104),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HUD_DIM, 1,
    )
