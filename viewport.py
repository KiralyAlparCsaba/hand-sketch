"""Pan/zoom transform between screen and canvas coordinates.

Phase 3: identity (offset=0, zoom=1). Pan/zoom logic gets wired up in
phase 5 (VIEW mode). The Viewport is the only thing that knows how canvas
coords map onto the screen, so adding pan/zoom later won't touch any
stroke data.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Viewport:
    offset_x: float = 0.0
    offset_y: float = 0.0
    zoom: float = 1.0

    def screen_to_canvas(self, sx: float, sy: float) -> Tuple[float, float]:
        return ((sx - self.offset_x) / self.zoom,
                (sy - self.offset_y) / self.zoom)

    def canvas_to_screen(self, cx: float, cy: float) -> Tuple[float, float]:
        return (cx * self.zoom + self.offset_x,
                cy * self.zoom + self.offset_y)

    def canvas_polyline_to_screen(self, pts: np.ndarray) -> np.ndarray:
        """Vectorized canvas->screen for a (N, 2) array of points."""
        return pts * self.zoom + np.array(
            [self.offset_x, self.offset_y], dtype=pts.dtype
        )

    def pan(self, dx: float, dy: float) -> None:
        self.offset_x += dx
        self.offset_y += dy

    def zoom_at(self, factor: float, screen_anchor: Tuple[float, float]) -> None:
        """Zoom by `factor`, keeping the canvas point under screen_anchor
        fixed on screen. Prevents the canvas from "flying off" when zooming."""
        ax, ay = screen_anchor
        cx, cy = self.screen_to_canvas(ax, ay)
        self.zoom *= factor
        self.offset_x = ax - cx * self.zoom
        self.offset_y = ay - cy * self.zoom
