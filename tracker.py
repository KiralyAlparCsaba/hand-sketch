"""MediaPipe hand tracking wrapper.

Returns hand landmarks per frame (or None if no hand detected). Also exposes
the standard MediaPipe hand connection list so renderers can draw the skeleton.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# (start_landmark_idx, end_landmark_idx) pairs for drawing the hand skeleton.
# Mirrors MediaPipe's HAND_CONNECTIONS.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


@dataclass
class HandLandmarks:
    """One frame's worth of hand landmark data."""
    pixels: np.ndarray      # shape (21, 2), int32 - pixel coords on the frame
    normalized: np.ndarray  # shape (21, 3), float32 - MediaPipe's xyz, x/y in [0,1]
    handedness: str         # "Left" or "Right" as labeled by MediaPipe


class HandTracker:
    """Thin wrapper over mp.solutions.hands.

    Configured for video (not static images), single-hand by default.
    Call .process(frame_bgr) per frame; call .close() at shutdown.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[HandLandmarks]:
        """Detect a hand in the given BGR frame. Returns None if no hand found."""
        h, w = frame_bgr.shape[:2]
        # MediaPipe wants RGB.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Marking the frame non-writeable is a small perf hint to MediaPipe.
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        normalized = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32
        )
        pixels = np.column_stack([
            np.clip(normalized[:, 0] * w, 0, w - 1).astype(np.int32),
            np.clip(normalized[:, 1] * h, 0, h - 1).astype(np.int32),
        ])

        handedness = "Unknown"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        return HandLandmarks(
            pixels=pixels, normalized=normalized, handedness=handedness
        )

    def close(self) -> None:
        self._hands.close()
