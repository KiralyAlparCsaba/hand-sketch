"""Document data model: strokes, groups, and the document itself.

Coordinate convention: ALL stroke points are stored in CANVAS coords.
Conversion to screen coords happens at render time via the Viewport. That
way pan/zoom never touches the underlying data — it only changes how the
existing data is mapped onto the screen.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# A point in 2D canvas space.
Point = Tuple[float, float]


@dataclass
class Stroke:
    """One continuous pen-down stroke. Points stored in canvas coords."""
    points: List[Point] = field(default_factory=list)

    def add_point(self, p: Point) -> None:
        self.points.append(p)

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class Group:
    """A drawing session — every stroke made between entering DRAW mode
    and leaving it. Treated as one object for grab/hit-test purposes."""
    strokes: List[Stroke] = field(default_factory=list)

    def is_empty(self) -> bool:
        return all(len(s) == 0 for s in self.strokes)


@dataclass
class Document:
    """All committed groups, plus the in-progress group/stroke (if any).

    Mutation API (used by modes.py):
      start_group / start_stroke / add_point / end_stroke / end_group

    Query API (used by render.py and grab mode):
      all_groups (incl. in-progress, for live render)
      hit_test_group (committed groups only, for grab)
    """
    groups: List[Group] = field(default_factory=list)
    current_group: Optional[Group] = None
    current_stroke: Optional[Stroke] = None

    # ---- mutation ----

    def start_group(self) -> None:
        self.current_group = Group()

    def end_group(self) -> None:
        if self.current_group is not None and not self.current_group.is_empty():
            self.groups.append(self.current_group)
        self.current_group = None

    def start_stroke(self) -> None:
        if self.current_group is None:
            # Defensive: shouldn't normally happen, but recover gracefully.
            self.start_group()
        self.current_stroke = Stroke()
        assert self.current_group is not None
        self.current_group.strokes.append(self.current_stroke)

    def end_stroke(self) -> None:
        # Drop an empty stroke (caused by gesture flicker — enter+exit DRAW
        # in fewer frames than it takes to add a point).
        if (
            self.current_stroke is not None
            and len(self.current_stroke) == 0
            and self.current_group is not None
            and self.current_group.strokes
            and self.current_group.strokes[-1] is self.current_stroke
        ):
            self.current_group.strokes.pop()
        self.current_stroke = None

    def add_point(self, p: Point) -> None:
        if self.current_stroke is None:
            return
        self.current_stroke.add_point(p)

    # ---- queries ----

    def all_groups(self) -> List[Group]:
        """Committed groups plus the in-progress one (so it renders live)."""
        out: List[Group] = list(self.groups)
        if self.current_group is not None and not self.current_group.is_empty():
            out.append(self.current_group)
        return out

    def erase_at(self, canvas_pt: Point, radius: float) -> None:
        """Pixel-style eraser: drop every point within `radius` of canvas_pt
        from every committed stroke. A stroke that's cut in the middle is
        split into two strokes (the surviving prefix and suffix). Strokes
        that drop below 2 points are removed; groups with no strokes left
        are removed. In-progress strokes are not touched."""
        tx, ty = canvas_pt
        r_sq = radius * radius
        new_groups: List[Group] = []
        for group in self.groups:
            new_strokes: List[Stroke] = []
            for stroke in group.strokes:
                run: List[Point] = []
                for px, py in stroke.points:
                    if (px - tx) * (px - tx) + (py - ty) * (py - ty) <= r_sq:
                        # Inside eraser: flush the surviving run, start fresh.
                        if len(run) >= 2:
                            new_strokes.append(Stroke(points=run))
                        run = []
                    else:
                        run.append((px, py))
                if len(run) >= 2:
                    new_strokes.append(Stroke(points=run))
            if new_strokes:
                new_groups.append(Group(strokes=new_strokes))
        self.groups = new_groups

    def hit_test_group(
        self, canvas_pt: Point, threshold: float
    ) -> Optional[Group]:
        """Return the committed group whose nearest segment is closest to
        canvas_pt, provided that distance is within `threshold` (canvas units).
        Used by GRAB mode."""
        target = np.array(canvas_pt, dtype=np.float32)
        best_group: Optional[Group] = None
        best_dist = float("inf")
        for group in self.groups:
            for stroke in group.strokes:
                if len(stroke) < 2:
                    continue
                pts = np.array(stroke.points, dtype=np.float32)
                d = _min_point_to_polyline(target, pts)
                if d < best_dist:
                    best_dist = d
                    best_group = group
        if best_dist <= threshold:
            return best_group
        return None


def _min_point_to_polyline(p: np.ndarray, polyline: np.ndarray) -> float:
    """Minimum distance from p to any segment of the polyline.
    polyline: (N, 2) array of vertices, N >= 2."""
    a = polyline[:-1]                          # (N-1, 2) segment starts
    b = polyline[1:]                           # (N-1, 2) segment ends
    ab = b - a
    ap = p - a
    seg_sq = np.maximum(np.sum(ab * ab, axis=1), 1e-12)
    t = np.clip(np.sum(ap * ab, axis=1) / seg_sq, 0.0, 1.0)
    closest = a + ab * t[:, None]
    diff = p - closest
    return float(np.min(np.sqrt(np.sum(diff * diff, axis=1))))
