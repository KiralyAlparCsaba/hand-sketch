"""Gesture classification from MediaPipe hand landmarks.

Two layers:
- classify_raw(landmarks): pure, per-frame classifier returning a Gesture enum
  plus diagnostic features for the HUD.
- GestureStateMachine: debounces raw classifications across N frames so a
  single misclassified frame can't flip modes. Emits exit/enter transitions
  on the frame a switch happens.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from tracker import HandLandmarks


class Gesture(Enum):
    NONE = "none"
    DRAW = "draw"
    PINCH = "pinch"
    PALM = "palm"


# MediaPipe landmark indices we use.
WRIST = 0
THUMB_TIP = 4
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_PIP = 14
RING_TIP = 16
PINKY_PIP = 18
PINKY_TIP = 20

# (pip_idx, tip_idx) pairs for the four non-thumb fingers.
FINGER_JOINTS = [
    (INDEX_PIP, INDEX_TIP),
    (MIDDLE_PIP, MIDDLE_TIP),
    (RING_PIP, RING_TIP),
    (PINKY_PIP, PINKY_TIP),
]

# Tunable thresholds. Watch the HUD in main.py and adjust if needed.
PINCH_THRESHOLD = 0.30   # ratio: (thumb-tip to index-tip dist) / hand size
EXTENSION_MARGIN = 1.05  # tip must be this much farther from wrist than PIP


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _hand_size(pts: np.ndarray) -> float:
    """A stable, scale-independent unit: wrist to middle-MCP distance."""
    return _dist(pts[WRIST], pts[MIDDLE_MCP])


def _is_extended(pts: np.ndarray, pip_idx: int, tip_idx: int) -> bool:
    """A non-thumb finger is extended if its tip is farther from the wrist
    than its PIP joint by some safety margin. Orientation-independent."""
    wrist = pts[WRIST]
    return _dist(wrist, pts[tip_idx]) > _dist(wrist, pts[pip_idx]) * EXTENSION_MARGIN


def _pinch_distance(pts: np.ndarray) -> float:
    """Thumb-tip to index-tip distance, normalized by hand size."""
    size = _hand_size(pts)
    if size < 1e-6:
        return float("inf")
    return _dist(pts[THUMB_TIP], pts[INDEX_TIP]) / size


@dataclass
class GestureFeatures:
    """Per-frame diagnostics for HUD/tuning. None when no hand is visible."""
    pinch_distance: float
    index_extended: bool
    middle_extended: bool
    ring_extended: bool
    pinky_extended: bool


def classify_raw(
    landmarks: Optional[HandLandmarks],
) -> Tuple[Gesture, Optional[GestureFeatures]]:
    """Per-frame gesture classification from raw landmarks.

    Uses normalized landmarks so the classifier is resolution-independent.
    Returns (Gesture, features) — features is None if no hand.
    """
    if landmarks is None:
        return Gesture.NONE, None

    pts = landmarks.normalized[:, :2]  # x, y only; z unused for now.

    index_ext, middle_ext, ring_ext, pinky_ext = [
        _is_extended(pts, pip, tip) for pip, tip in FINGER_JOINTS
    ]
    pinch_d = _pinch_distance(pts)

    features = GestureFeatures(
        pinch_distance=pinch_d,
        index_extended=index_ext,
        middle_extended=middle_ext,
        ring_extended=ring_ext,
        pinky_extended=pinky_ext,
    )

    # Order matters. PINCH wins over other patterns since a pinching hand can
    # have arbitrary other-finger states. Then DRAW (one-finger-up), then PALM.
    if pinch_d < PINCH_THRESHOLD:
        return Gesture.PINCH, features

    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return Gesture.DRAW, features

    if index_ext and middle_ext and ring_ext and pinky_ext:
        return Gesture.PALM, features

    return Gesture.NONE, features


# Transition kind: "exit" or "enter".
Transition = Tuple[str, Gesture]


class GestureStateMachine:
    """Debounces raw gestures: only switches mode after `debounce_frames`
    consecutive frames of the new gesture. Emits exit/enter transitions on
    the frame the switch happens."""

    def __init__(self, debounce_frames: int = 4):
        self._debounce = debounce_frames
        self._current = Gesture.NONE
        self._candidate = Gesture.NONE
        self._candidate_count = 0

    @property
    def current(self) -> Gesture:
        return self._current

    def update(self, raw: Gesture) -> Tuple[Gesture, List[Transition]]:
        if raw == self._current:
            # Stable. Cancel any pending change.
            self._candidate = raw
            self._candidate_count = 0
            return self._current, []

        # raw differs from current.
        if raw == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = raw
            self._candidate_count = 1

        if self._candidate_count >= self._debounce:
            old = self._current
            self._current = self._candidate
            self._candidate_count = 0
            return self._current, [("exit", old), ("enter", self._current)]

        return self._current, []
