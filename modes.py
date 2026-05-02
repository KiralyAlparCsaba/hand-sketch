"""Mode handlers — the bridge between gesture transitions and document/
viewport mutations.

One handler per mode. The main loop calls all of them every frame; each
handler reacts only to the transitions and current state it cares about.
This keeps each mode's logic self-contained.

Phase 3: DrawHandler is implemented; GrabHandler and ViewHandler are
stubs to be filled in by phases 4 and 5.
"""

from typing import List, Optional

import numpy as np

from document import Document, Group, Point
from gestures import Gesture, INDEX_TIP, THUMB_TIP, Transition
from tracker import HandLandmarks
from viewport import Viewport


# Eraser radius in canvas units. Imported by render.py so the cursor
# circle visually matches the actual erase area.
ERASER_RADIUS = 30.0

# Maximum distance (canvas units) from the pinch midpoint to a stroke for
# a grab to register. Imported by render.py for the pinch indicator ring.
GRAB_HIT_THRESHOLD = 30.0


class EMAFilter2D:
    """Exponential moving average for a 2D point.

    alpha=1.0 -> no smoothing (raw input)
    alpha=0.5 -> moderate (default)
    alpha=0.2 -> heavy smoothing (laggy but smooth)

    Reset between strokes so a new stroke doesn't get pulled toward the
    previous stroke's end position.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._value: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._value = None

    def update(self, p: np.ndarray) -> np.ndarray:
        if self._value is None:
            self._value = p.astype(np.float32).copy()
        else:
            self._value = self.alpha * p + (1.0 - self.alpha) * self._value
        return self._value.copy()


class DrawHandler:
    """Manages stroke creation while in DRAW mode.

    On entering DRAW: start a new group + first stroke; reset smoothing.
    Each frame in DRAW: append the smoothed index fingertip in canvas coords.
    On exiting DRAW: end the stroke and group (closing the session).
    """

    def __init__(self, smoothing_alpha: float = 0.5):
        self._fingertip_filter = EMAFilter2D(alpha=smoothing_alpha)

    def handle(
        self,
        mode: Gesture,
        raw: Gesture,
        transitions: List[Transition],
        landmarks: Optional[HandLandmarks],
        document: Document,
        viewport: Viewport,
    ) -> None:
        for kind, gesture in transitions:
            if gesture != Gesture.DRAW:
                continue
            if kind == "exit":
                document.end_stroke()
                document.end_group()
                self._fingertip_filter.reset()
            elif kind == "enter":
                document.start_group()
                document.start_stroke()
                self._fingertip_filter.reset()

        # Append points only when BOTH the debounced mode and the raw
        # per-frame classification say DRAW. The state-machine debounce
        # gives us stable stroke boundaries; the raw check stops us from
        # tracking the finger through a pose transition (the "smush").
        if mode == Gesture.DRAW and raw == Gesture.DRAW and landmarks is not None:
            tip_screen = landmarks.pixels[INDEX_TIP].astype(np.float32)
            tip_smooth = self._fingertip_filter.update(tip_screen)
            cx, cy = viewport.screen_to_canvas(
                float(tip_smooth[0]), float(tip_smooth[1])
            )
            document.add_point((cx, cy))


class EraseHandler:
    """Eraser tool. While in ERASE mode, the index fingertip deletes any
    group whose nearest stroke segment comes within `eraser_radius` of it.

    Symmetric with DrawHandler: requires both `mode == ERASE` and
    `raw == ERASE` to actually delete, so a pose transition into ERASE
    can't grab strokes underneath as the hand moves.
    """

    def __init__(self, eraser_radius: float = ERASER_RADIUS):
        self._radius = eraser_radius

    def handle(
        self,
        mode: Gesture,
        raw: Gesture,
        transitions: List[Transition],
        landmarks: Optional[HandLandmarks],
        document: Document,
        viewport: Viewport,
    ) -> None:
        if (
            mode == Gesture.ERASE
            and raw == Gesture.ERASE
            and landmarks is not None
        ):
            tip_screen = landmarks.pixels[INDEX_TIP].astype(np.float32)
            cx, cy = viewport.screen_to_canvas(
                float(tip_screen[0]), float(tip_screen[1])
            )
            document.erase_at((cx, cy), self._radius)


class GrabHandler:
    """PINCH-to-grab: on entering PINCH, hit-test for the nearest committed
    group. While the pinch is held, translate that group so its initial
    pinch point follows the hand.

    The translation works on a SNAPSHOT of the group's points taken at grab
    time — every frame we recompute `new = original + delta`, never
    incrementally translate. This avoids floating-point drift over long
    drags.

    The pinch midpoint is EMA-smoothed so the grabbed group doesn't shake
    with hand jitter.
    """

    def __init__(
        self,
        hit_threshold: float = GRAB_HIT_THRESHOLD,
        smoothing_alpha: float = 0.5,
    ):
        self._hit_threshold = hit_threshold
        self._pinch_filter = EMAFilter2D(alpha=smoothing_alpha)
        self._grabbed_group: Optional[Group] = None
        self._initial_pinch_canvas: Optional[Tuple[float, float]] = None
        self._original_points: Optional[List[List[Point]]] = None

    def handle(
        self,
        mode: Gesture,
        raw: Gesture,
        transitions: List[Transition],
        landmarks: Optional[HandLandmarks],
        document: Document,
        viewport: Viewport,
    ) -> None:
        for kind, gesture in transitions:
            if gesture != Gesture.PINCH:
                continue
            if kind == "exit":
                self._release()
            elif kind == "enter":
                self._try_grab(landmarks, document, viewport)

        if (
            mode == Gesture.PINCH
            and raw == Gesture.PINCH
            and landmarks is not None
            and self._grabbed_group is not None
        ):
            self._update_position(landmarks, viewport)

    # ---- internals ----

    def _pinch_midpoint_smoothed(self, landmarks: HandLandmarks) -> np.ndarray:
        thumb = landmarks.pixels[THUMB_TIP].astype(np.float32)
        index = landmarks.pixels[INDEX_TIP].astype(np.float32)
        mid = (thumb + index) * 0.5
        return self._pinch_filter.update(mid)

    def _try_grab(
        self,
        landmarks: Optional[HandLandmarks],
        document: Document,
        viewport: Viewport,
    ) -> None:
        if landmarks is None:
            return
        self._pinch_filter.reset()
        mid_screen = self._pinch_midpoint_smoothed(landmarks)
        cx, cy = viewport.screen_to_canvas(
            float(mid_screen[0]), float(mid_screen[1])
        )
        group = document.hit_test_group((cx, cy), self._hit_threshold)
        if group is None:
            return
        self._grabbed_group = group
        self._initial_pinch_canvas = (cx, cy)
        # Snapshot original positions so per-frame translate doesn't drift.
        self._original_points = [
            list(stroke.points) for stroke in group.strokes
        ]

    def _release(self) -> None:
        self._grabbed_group = None
        self._initial_pinch_canvas = None
        self._original_points = None
        self._pinch_filter.reset()

    def _update_position(
        self, landmarks: HandLandmarks, viewport: Viewport
    ) -> None:
        assert self._grabbed_group is not None
        assert self._initial_pinch_canvas is not None
        assert self._original_points is not None

        mid_screen = self._pinch_midpoint_smoothed(landmarks)
        cx, cy = viewport.screen_to_canvas(
            float(mid_screen[0]), float(mid_screen[1])
        )
        ix, iy = self._initial_pinch_canvas
        dx = cx - ix
        dy = cy - iy
        for stroke, original in zip(
            self._grabbed_group.strokes, self._original_points
        ):
            stroke.points = [(x + dx, y + dy) for x, y in original]


class ViewHandler:
    """Stub — implemented in phase 5 (PALM-to-pan/zoom)."""

    def handle(
        self,
        mode: Gesture,
        raw: Gesture,
        transitions: List[Transition],
        landmarks: Optional[HandLandmarks],
        document: Document,
        viewport: Viewport,
    ) -> None:
        pass
