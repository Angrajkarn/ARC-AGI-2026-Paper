"""
ArcObject — rich representation of a detected object within an ARC grid.

Objects are connected regions of the same colour (or multi-colour shapes
discovered by higher-level detectors).  Every object carries geometry,
appearance and relational metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass(frozen=False)
class BoundingBox:
    """Axis-aligned bounding box."""

    row_min: int
    row_max: int
    col_min: int
    col_max: int

    @property
    def height(self) -> int:
        return self.row_max - self.row_min + 1

    @property
    def width(self) -> int:
        return self.col_max - self.col_min + 1

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def center(self) -> Tuple[float, float]:
        return (self.row_min + self.row_max) / 2, (self.col_min + self.col_max) / 2

    def contains(self, row: int, col: int) -> bool:
        return self.row_min <= row <= self.row_max and self.col_min <= col <= self.col_max

    def overlaps(self, other: "BoundingBox") -> bool:
        return (
            self.row_min <= other.row_max
            and self.row_max >= other.row_min
            and self.col_min <= other.col_max
            and self.col_max >= other.col_min
        )

    def __repr__(self) -> str:
        return f"BBox(r={self.row_min}:{self.row_max}, c={self.col_min}:{self.col_max})"


@dataclass
class ArcObject:
    """Represents a detected object within an ARC grid.

    Attributes:
        object_id:     Unique identifier within the grid.
        color:         Primary colour (integer 0-9).
        pixels:        Set of (row, col) coordinates.
        bounding_box:  Tight axis-aligned bounding box.
        is_rectangular: True if the pixel set fills its bounding box.
        is_line:       True if height==1 or width==1.
        is_symmetric_h: Horizontal (left-right) symmetry.
        is_symmetric_v: Vertical (top-bottom) symmetry.
        holes:         List of pixel-sets representing enclosed holes.
        tags:          Arbitrary string tags for higher-level detectors.
        properties:    Key→value store for computed properties.
    """

    object_id: int
    color: int
    pixels: Set[Tuple[int, int]]
    bounding_box: BoundingBox
    is_rectangular: bool = False
    is_line: bool = False
    is_symmetric_h: bool = False
    is_symmetric_v: bool = False
    holes: List[Set[Tuple[int, int]]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------

    @property
    def area(self) -> int:
        """Number of pixels in the object."""
        return len(self.pixels)

    @property
    def perimeter(self) -> int:
        """Approximate perimeter (4-connected boundary pixels)."""
        count = 0
        for r, c in self.pixels:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if (r + dr, c + dc) not in self.pixels:
                    count += 1
        return count

    @property
    def center(self) -> Tuple[float, float]:
        """Centroid of the pixel set."""
        rows = [r for r, _ in self.pixels]
        cols = [c for _, c in self.pixels]
        return sum(rows) / len(rows), sum(cols) / len(cols)

    @property
    def orientation(self) -> str:
        """Rough orientation: 'horizontal', 'vertical', or 'square'."""
        h = self.bounding_box.height
        w = self.bounding_box.width
        if h > w * 1.5:
            return "vertical"
        if w > h * 1.5:
            return "horizontal"
        return "square"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pixels(
        cls,
        object_id: int,
        color: int,
        pixels: Set[Tuple[int, int]],
    ) -> "ArcObject":
        """Build an ArcObject from a pixel set, computing derived fields.

        Args:
            object_id: Unique id.
            color:     Object colour.
            pixels:    Set of (row, col) coordinates.

        Returns:
            Fully initialised ArcObject.
        """
        if not pixels:
            raise ValueError("Cannot create ArcObject from empty pixel set")

        rows = [r for r, _ in pixels]
        cols = [c for _, c in pixels]
        bbox = BoundingBox(min(rows), max(rows), min(cols), max(cols))

        is_rect = (len(pixels) == bbox.area)
        is_line = (bbox.height == 1 or bbox.width == 1)

        # Symmetry checks (translate to local coords first)
        local = {(r - bbox.row_min, c - bbox.col_min) for r, c in pixels}
        H, W = bbox.height, bbox.width

        is_sym_h = all((r, W - 1 - c) in local for r, c in local)
        is_sym_v = all((H - 1 - r, c) in local for r, c in local)

        return cls(
            object_id=object_id,
            color=color,
            pixels=pixels,
            bounding_box=bbox,
            is_rectangular=is_rect,
            is_line=is_line,
            is_symmetric_h=is_sym_h,
            is_symmetric_v=is_sym_v,
        )

    # ------------------------------------------------------------------
    # Spatial transforms (return new objects, don't mutate)
    # ------------------------------------------------------------------

    def translate(self, dr: int, dc: int) -> "ArcObject":
        """Shift all pixels by (dr, dc)."""
        new_pixels = {(r + dr, c + dc) for r, c in self.pixels}
        return ArcObject.from_pixels(self.object_id, self.color, new_pixels)

    def mirror_horizontal(self, axis_col: Optional[int] = None) -> "ArcObject":
        """Mirror the object horizontally about axis_col (default: object centre)."""
        if axis_col is None:
            axis_col = round(self.bounding_box.center[1])
        new_pixels = {(r, 2 * axis_col - c) for r, c in self.pixels}
        return ArcObject.from_pixels(self.object_id, self.color, new_pixels)

    def rotate_90(self, times: int = 1) -> "ArcObject":
        """Rotate the object 90° clockwise *times* times (in local coords)."""
        # Work in local coordinates then re-offset
        local = {(r - self.bounding_box.row_min, c - self.bounding_box.col_min)
                 for r, c in self.pixels}
        H, W = self.bounding_box.height, self.bounding_box.width
        for _ in range(times % 4):
            local = {(c, H - 1 - r) for r, c in local}
            H, W = W, H

        # Re-apply original offset
        new_pixels = {(r + self.bounding_box.row_min, c + self.bounding_box.col_min)
                      for r, c in local}
        return ArcObject.from_pixels(self.object_id, self.color, new_pixels)

    def recolor(self, new_color: int) -> "ArcObject":
        """Return a copy of this object with a different colour."""
        obj = ArcObject.from_pixels(self.object_id, new_color, set(self.pixels))
        obj.holes = [set(h) for h in self.holes]
        obj.tags = list(self.tags)
        obj.properties = dict(self.properties)
        return obj

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def to_grid_snippet(self) -> "np.ndarray":
        """Return a minimal 2-D numpy array cropped to the bounding box."""
        H = self.bounding_box.height
        W = self.bounding_box.width
        arr = np.zeros((H, W), dtype=np.int8)
        for r, c in self.pixels:
            arr[r - self.bounding_box.row_min, c - self.bounding_box.col_min] = self.color
        return arr

    def distance_to(self, other: "ArcObject") -> float:
        """Euclidean distance between centroids."""
        r1, c1 = self.center
        r2, c2 = other.center
        return math.hypot(r2 - r1, c2 - c1)

    def is_adjacent_to(self, other: "ArcObject", diagonal: bool = False) -> bool:
        """Return True if this object touches *other* (4-connected by default)."""
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diagonal:
            offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for r, c in self.pixels:
            for dr, dc in offsets:
                if (r + dr, c + dc) in other.pixels:
                    return True
        return False

    def contains_object(self, other: "ArcObject") -> bool:
        """Return True if all pixels of *other* are inside this object's bbox."""
        b = self.bounding_box
        o = other.bounding_box
        return (
            b.row_min <= o.row_min
            and b.row_max >= o.row_max
            and b.col_min <= o.col_min
            and b.col_max >= o.col_max
        )

    def __repr__(self) -> str:
        return (
            f"ArcObject(id={self.object_id}, color={self.color}, "
            f"area={self.area}, bbox={self.bounding_box})"
        )


__all__ = ["ArcObject", "BoundingBox"]
