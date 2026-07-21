# Auto-generated Kaggle Submission Script for ARC-AGI-2026 Engine
# Environment: Offline Python 3.10+
import os, sys, json, time
import numpy as np

# --- Begin src/core/grid/grid.py ---
"""
ArcGrid — core grid data structure for ARC-AGI-2 tasks.

Grids are the fundamental unit of the ARC format.  Rather than treating
them as raw NumPy tensors we wrap them in a rich dataclass that carries
derived metadata (size, color set, background) and convenience methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np


# ARC colour palette: integers 0-9 map to named colours for readability.
COLOR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "azure",
    9: "maroon",
}


@dataclass
class ArcGrid:
    """Rich representation of a single ARC grid.

    Attributes:
        pixels:     2-D NumPy array of shape (height, width), dtype int8, values 0-9.
        background: The most-common (or explicitly set) background colour.
        metadata:   Arbitrary key→value store for task-level annotations.
    """

    pixels: np.ndarray  # shape (H, W), dtype int8
    background: int = 0
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived properties (computed once on first access via property)
    # ------------------------------------------------------------------

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def colors(self) -> Set[int]:
        """Set of unique colour values present in the grid."""
        return set(np.unique(self.pixels).tolist())

    @property
    def non_background_colors(self) -> Set[int]:
        return self.colors - {self.background}

    @property
    def non_background_ratio(self) -> float:
        """Fraction of pixels that are not background."""
        non_bg = int((self.pixels != self.background).sum())
        return non_bg / max(self.num_pixels, 1)

    @property
    def size(self) -> Tuple[int, int]:
        """(height, width) tuple."""
        return (self.height, self.width)

    @property
    def num_pixels(self) -> int:
        return self.height * self.width

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_list(
        cls,
        grid: List[List[int]],
        background: Optional[int] = None,
    ) -> "ArcGrid":
        """Create an ArcGrid from a 2-D Python list (ARC JSON format).

        Args:
            grid:       2-D list of ints 0-9.
            background: Override background colour detection.

        Returns:
            ArcGrid instance.
        """
        arr = np.array(grid, dtype=np.int8)
        if background is None:
            # Detect background as most-frequent colour
            values, counts = np.unique(arr, return_counts=True)
            background = int(values[np.argmax(counts)])
        return cls(pixels=arr, background=background)

    @classmethod
    def empty(cls, height: int, width: int, fill: int = 0) -> "ArcGrid":
        """Create a blank grid filled with *fill* colour."""
        return cls(pixels=np.full((height, width), fill, dtype=np.int8), background=fill)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, background: int = 0) -> "ArcGrid":
        """Wrap a NumPy array as an ArcGrid."""
        return cls(pixels=arr.astype(np.int8), background=background)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_list(self) -> List[List[int]]:
        """Convert back to ARC JSON format (2-D list of ints)."""
        return self.pixels.tolist()

    def to_numpy(self) -> np.ndarray:
        """Return a copy of the underlying NumPy array."""
        return self.pixels.copy()

    # ------------------------------------------------------------------
    # Equality & hashing
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArcGrid):
            return NotImplemented
        return (
            self.pixels.shape == other.pixels.shape
            and bool(np.array_equal(self.pixels, other.pixels))
        )

    def __hash__(self) -> int:
        return hash(self.pixels.tobytes())

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> "ArcGrid":
        """Return a deep copy of the grid."""
        return ArcGrid(
            pixels=self.pixels.copy(),
            background=self.background,
            metadata=self.metadata.copy(),
        )

    # ------------------------------------------------------------------
    # Pixel access helpers
    # ------------------------------------------------------------------

    def get(self, row: int, col: int) -> int:
        """Get the colour at (row, col)."""
        return int(self.pixels[row, col])

    def set(self, row: int, col: int, color: int) -> None:
        """Set the colour at (row, col) in-place."""
        self.pixels[row, col] = color

    def in_bounds(self, row: int, col: int) -> bool:
        """Check whether (row, col) is within the grid."""
        return 0 <= row < self.height and 0 <= col < self.width

    def pixels_of_color(self, color: int) -> List[Tuple[int, int]]:
        """Return all (row, col) positions with the given colour."""
        rows, cols = np.where(self.pixels == color)
        return list(zip(rows.tolist(), cols.tolist()))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ArcGrid(height={self.height}, width={self.width}, "
            f"colors={sorted(self.colors)}, background={self.background})"
        )

    def __str__(self) -> str:
        rows = []
        for row in self.pixels.tolist():
            rows.append(" ".join(str(c) for c in row))
        return "\n".join(rows)


__all__ = ["ArcGrid", "COLOR_NAMES"]

# --- End src/core/grid/grid.py ---

# --- Begin src/core/objects/arc_object.py ---
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

# --- End src/core/objects/arc_object.py ---

# --- Begin src/core/objects/detector.py ---
"""
ObjectDetector — detects and segments objects in ARC grids.

Detection strategies:
  - BFS / DFS connected components (4-connected or 8-connected)
  - Union-Find (Disjoint Set Union) connected components
  - Flood fill
  - Rectangle detection
  - Line detection
  - Border / frame detection
  - Hole detection (interior enclosed regions)
  - Noise pixel detection
  - Symmetric object detection
  - Repeated pattern detection
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np



# ---------------------------------------------------------------------------
# Union-Find (DSU)
# ---------------------------------------------------------------------------

class _DSU:
    """Disjoint Set Union with path compression and union by rank."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# ---------------------------------------------------------------------------
# ObjectDetector
# ---------------------------------------------------------------------------

class ObjectDetector:
    """Detects objects within ARC grids using multiple strategies.

    Usage::

        detector = ObjectDetector(connectivity=4)
        objects = detector.detect(grid)
    """

    def __init__(
        self,
        connectivity: int = 4,
        min_size: int = 1,
        ignore_background: bool = True,
    ) -> None:
        """
        Args:
            connectivity:      4 (edge) or 8 (edge+diagonal) connected.
            min_size:          Minimum object area to keep.
            ignore_background: If True, background pixels are not grouped.
        """
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        self.connectivity = connectivity
        self.min_size = min_size
        self.ignore_background = ignore_background

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, grid: ArcGrid) -> List[ArcObject]:
        """Detect all objects in a grid using BFS connected components.

        Args:
            grid: The grid to analyse.

        Returns:
            List of ArcObject instances, ordered by area descending.
        """
        objects = self._bfs_components(grid)
        objects = [o for o in objects if o.area >= self.min_size]

        # Enrich with secondary detections
        self._detect_holes(objects, grid)
        self._tag_special(objects, grid)

        objects.sort(key=lambda o: o.area, reverse=True)
        return objects

    def detect_with_dsu(self, grid: ArcGrid) -> List[ArcObject]:
        """Alternative detector using Union-Find for connected components.

        Args:
            grid: The grid to analyse.

        Returns:
            List of ArcObject instances.
        """
        H, W = grid.height, grid.width
        N = H * W
        dsu = _DSU(N)

        def idx(r: int, c: int) -> int:
            return r * W + c

        offsets = self._offsets()
        for r in range(H):
            for c in range(W):
                color = grid.get(r, c)
                if self.ignore_background and color == grid.background:
                    continue
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and grid.get(nr, nc) == color:
                        dsu.union(idx(r, c), idx(nr, nc))

        # Group pixels by root representative
        groups: Dict[int, Set[Tuple[int, int]]] = {}
        colors: Dict[int, int] = {}
        for r in range(H):
            for c in range(W):
                color = grid.get(r, c)
                if self.ignore_background and color == grid.background:
                    continue
                root = dsu.find(idx(r, c))
                groups.setdefault(root, set()).add((r, c))
                colors[root] = color

        objects = []
        for oid, (root, pixels) in enumerate(groups.items()):
            if len(pixels) < self.min_size:
                continue
            objects.append(ArcObject.from_pixels(oid, colors[root], pixels))

        objects.sort(key=lambda o: o.area, reverse=True)
        return objects

    def flood_fill(
        self,
        grid: ArcGrid,
        start_row: int,
        start_col: int,
        target_color: Optional[int] = None,
    ) -> Set[Tuple[int, int]]:
        """Run flood fill from (start_row, start_col) matching target_color.

        Args:
            grid:         The grid.
            start_row:    Starting row.
            start_col:    Starting column.
            target_color: Colour to fill from (defaults to colour at start).

        Returns:
            Set of (row, col) positions reachable.
        """
        if target_color is None:
            target_color = grid.get(start_row, start_col)

        visited: Set[Tuple[int, int]] = set()
        queue: deque = deque([(start_row, start_col)])
        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            if not grid.in_bounds(r, c):
                continue
            if grid.get(r, c) != target_color:
                continue
            visited.add((r, c))
            for dr, dc in self._offsets():
                queue.append((r + dr, c + dc))
        return visited

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bfs_components(self, grid: ArcGrid) -> List[ArcObject]:
        """BFS-based connected component labelling."""
        H, W = grid.height, grid.width
        visited = np.zeros((H, W), dtype=bool)
        objects: List[ArcObject] = []
        oid = 0

        for start_r in range(H):
            for start_c in range(W):
                color = grid.get(start_r, start_c)
                if visited[start_r, start_c]:
                    continue
                if self.ignore_background and color == grid.background:
                    visited[start_r, start_c] = True
                    continue

                # BFS
                pixels: Set[Tuple[int, int]] = set()
                queue: deque = deque([(start_r, start_c)])
                while queue:
                    r, c = queue.popleft()
                    if grid.get(r, c) != color:
                        continue
                    visited[r, c] = True
                    pixels.add((r, c))
                    for dr, dc in self._offsets():
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                            queue.append((nr, nc))

                if pixels:
                    objects.append(ArcObject.from_pixels(oid, color, pixels))
                    oid += 1

        return objects

    def _offsets(self) -> List[Tuple[int, int]]:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.connectivity == 8:
            offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return offsets

    def _detect_holes(self, objects: List[ArcObject], grid: ArcGrid) -> None:
        """Detect enclosed holes within each object (modifies in-place)."""
        H, W = grid.height, grid.width
        all_object_pixels: Set[Tuple[int, int]] = set()
        for obj in objects:
            all_object_pixels |= obj.pixels

        for obj in objects:
            bbox = obj.bounding_box
            # Search background pixels inside the bounding box
            interior_bg: Set[Tuple[int, int]] = set()
            for r in range(bbox.row_min, bbox.row_max + 1):
                for c in range(bbox.col_min, bbox.col_max + 1):
                    if (r, c) not in obj.pixels and grid.get(r, c) == grid.background:
                        interior_bg.add((r, c))

            if not interior_bg:
                continue

            # Flood fill from border — anything reachable is NOT a hole
            not_hole: Set[Tuple[int, int]] = set()
            for start in list(interior_bg):
                if start in not_hole:
                    continue
                # If this bg pixel can reach outside the bbox it's not a hole
                component: Set[Tuple[int, int]] = set()
                queue: deque = deque([start])
                escapes = False
                while queue:
                    r, c = queue.popleft()
                    if (r, c) in component:
                        continue
                    if not grid.in_bounds(r, c):
                        escapes = True
                        continue
                    if (r, c) in obj.pixels:
                        continue
                    if not bbox.contains(r, c):
                        escapes = True
                        continue
                    component.add((r, c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        queue.append((r + dr, c + dc))

                if escapes:
                    not_hole |= component
                else:
                    obj.holes.append(component)

    def _tag_special(self, objects: List[ArcObject], grid: ArcGrid) -> None:
        """Add descriptive tags to objects (rectangle, line, border, etc.)."""
        if not objects:
            return

        max_area = max(o.area for o in objects)
        min_area = min(o.area for o in objects)

        for obj in objects:
            if obj.is_rectangular:
                obj.tags.append("rectangle")
            if obj.is_line:
                obj.tags.append("line")
            if obj.holes:
                obj.tags.append("has_holes")
            if obj.is_symmetric_h:
                obj.tags.append("symmetric_h")
            if obj.is_symmetric_v:
                obj.tags.append("symmetric_v")
            if obj.area == max_area and len(objects) > 1:
                obj.tags.append("largest")
            if obj.area == min_area and len(objects) > 1:
                obj.tags.append("smallest")
            if obj.area == 1:
                obj.tags.append("noise")
            # Border: bounding box touches grid edge
            bbox = obj.bounding_box
            if (bbox.row_min == 0 or bbox.row_max == grid.height - 1
                    or bbox.col_min == 0 or bbox.col_max == grid.width - 1):
                obj.tags.append("border")

    def detect_rectangles(self, grid: ArcGrid) -> List[ArcObject]:
        """Return only rectangular objects."""
        return [o for o in self.detect(grid) if o.is_rectangular]

    def detect_lines(self, grid: ArcGrid) -> List[ArcObject]:
        """Return only line-shaped objects (1 row or 1 column wide)."""
        return [o for o in self.detect(grid) if o.is_line]

    def detect_noise(self, grid: ArcGrid) -> List[ArcObject]:
        """Return single-pixel noise objects."""
        return [o for o in self.detect(grid) if o.area == 1]


__all__ = ["ObjectDetector"]

# --- End src/core/objects/detector.py ---

# --- Begin src/dsl/primitives/transforms.py ---
"""
DSL Primitives — ~35 atomic grid transformation functions.

Each primitive is a pure function:
    (ArcGrid, **kwargs) → ArcGrid

All primitives are registered in PRIMITIVE_REGISTRY for the DSL executor.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

PRIMITIVE_REGISTRY: Dict[str, Callable[..., ArcGrid]] = {}


def _register(name: str):
    """Decorator to register a primitive function by name."""
    def decorator(fn: Callable) -> Callable:
        PRIMITIVE_REGISTRY[name] = fn
        return fn
    return decorator


# ------------------------------------------------------------------
# Rotation & Reflection
# ------------------------------------------------------------------

@_register("rotate_90")
def rotate_90(grid: ArcGrid, times: int = 1) -> ArcGrid:
    """Rotate grid 90° clockwise *times* times."""
    arr = grid.pixels.copy()
    for _ in range(times % 4):
        arr = np.rot90(arr, k=-1)  # k=-1 = 90° clockwise
    return ArcGrid(pixels=arr, background=grid.background)


@_register("rotate_180")
def rotate_180(grid: ArcGrid) -> ArcGrid:
    return rotate_90(grid, times=2)


@_register("rotate_270")
def rotate_270(grid: ArcGrid) -> ArcGrid:
    return rotate_90(grid, times=3)


@_register("mirror_horizontal")
def mirror_horizontal(grid: ArcGrid) -> ArcGrid:
    """Flip left-right."""
    return ArcGrid(pixels=np.fliplr(grid.pixels.copy()), background=grid.background)


@_register("mirror_vertical")
def mirror_vertical(grid: ArcGrid) -> ArcGrid:
    """Flip top-bottom."""
    return ArcGrid(pixels=np.flipud(grid.pixels.copy()), background=grid.background)


@_register("mirror_diagonal")
def mirror_diagonal(grid: ArcGrid) -> ArcGrid:
    """Transpose (mirror along main diagonal)."""
    return ArcGrid(pixels=grid.pixels.T.copy(), background=grid.background)


@_register("mirror_antidiagonal")
def mirror_antidiagonal(grid: ArcGrid) -> ArcGrid:
    """Mirror along the anti-diagonal."""
    return ArcGrid(pixels=np.rot90(np.fliplr(grid.pixels.copy()), k=1),
                   background=grid.background)


# ------------------------------------------------------------------
# Crop & Resize
# ------------------------------------------------------------------

@_register("crop")
def crop(grid: ArcGrid, row_min: int, row_max: int, col_min: int, col_max: int) -> ArcGrid:
    """Crop to a rectangular region (inclusive bounds)."""
    arr = grid.pixels[row_min:row_max + 1, col_min:col_max + 1].copy()
    return ArcGrid(pixels=arr, background=grid.background)


@_register("crop_to_content")
def crop_to_content(grid: ArcGrid) -> ArcGrid:
    """Crop to the tightest bounding box around non-background pixels."""
    rows, cols = np.where(grid.pixels != grid.background)
    if len(rows) == 0:
        return grid.copy()
    return crop(grid, int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max()))


@_register("scale")
def scale(grid: ArcGrid, factor: int) -> ArcGrid:
    """Upscale grid by integer factor (nearest-neighbour)."""
    arr = np.repeat(np.repeat(grid.pixels, factor, axis=0), factor, axis=1)
    return ArcGrid(pixels=arr, background=grid.background)


@_register("resize")
def resize(grid: ArcGrid, new_height: int, new_width: int) -> ArcGrid:
    """Resize grid to (new_height, new_width) via nearest-neighbour."""
    from scipy.ndimage import zoom
    arr = grid.pixels.astype(float)
    zy = new_height / grid.height
    zx = new_width / grid.width
    resized = zoom(arr, (zy, zx), order=0).astype(np.int8)
    resized = resized[:new_height, :new_width]  # guard against off-by-one
    return ArcGrid(pixels=resized, background=grid.background)


# ------------------------------------------------------------------
# Translation
# ------------------------------------------------------------------

@_register("translate")
def translate(grid: ArcGrid, dr: int, dc: int, fill: Optional[int] = None) -> ArcGrid:
    """Shift all non-background pixels by (dr, dc), filling vacated cells."""
    if fill is None:
        fill = grid.background
    H, W = grid.height, grid.width
    arr = np.full((H, W), fill, dtype=np.int8)
    for r in range(H):
        for c in range(W):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                arr[nr, nc] = grid.get(r, c)
    return ArcGrid(pixels=arr, background=grid.background)


# ------------------------------------------------------------------
# Repetition
# ------------------------------------------------------------------

@_register("tile")
def tile(grid: ArcGrid, rows: int, cols: int) -> ArcGrid:
    """Tile the grid into an (rows x cols) mosaic."""
    arr = np.tile(grid.pixels, (rows, cols))
    return ArcGrid(pixels=arr, background=grid.background)


@_register("repeat_pattern")
def repeat_pattern(grid: ArcGrid, direction: str = "right", times: int = 1) -> ArcGrid:
    """Repeat the grid content along a direction."""
    if direction == "right":
        arr = np.concatenate([grid.pixels] * (times + 1), axis=1)
    elif direction == "down":
        arr = np.concatenate([grid.pixels] * (times + 1), axis=0)
    else:
        raise ValueError(f"Unknown direction: {direction!r}")
    return ArcGrid(pixels=arr, background=grid.background)


# ------------------------------------------------------------------
# Colour Operations
# ------------------------------------------------------------------

@_register("replace_color")
def replace_color(grid: ArcGrid, source_color: int, target_color: int) -> ArcGrid:
    """Replace all pixels of source_color with target_color."""
    arr = grid.pixels.copy()
    arr[arr == source_color] = target_color
    bg = target_color if grid.background == source_color else grid.background
    return ArcGrid(pixels=arr, background=bg)


@_register("swap_colors")
def swap_colors(grid: ArcGrid, color_a: int, color_b: int) -> ArcGrid:
    """Swap two colours throughout the grid."""
    arr = grid.pixels.copy()
    mask_a = arr == color_a
    mask_b = arr == color_b
    arr[mask_a] = color_b
    arr[mask_b] = color_a
    return ArcGrid(pixels=arr, background=grid.background)


@_register("normalize_colors")
def normalize_colors(grid: ArcGrid) -> ArcGrid:
    """Remap colours to canonical form (most common → 0, next → 1, etc.)."""
    values, counts = np.unique(grid.pixels, return_counts=True)
    sorted_vals = values[np.argsort(-counts)]
    mapping = {int(old): new for new, old in enumerate(sorted_vals)}
    arr = np.vectorize(lambda x: mapping[x])(grid.pixels).astype(np.int8)
    return ArcGrid(pixels=arr, background=mapping[grid.background])


# ------------------------------------------------------------------
# Fill & Flood Operations
# ------------------------------------------------------------------

@_register("flood_fill")
def flood_fill(grid: ArcGrid, row: int, col: int, new_color: int) -> ArcGrid:
    """Flood fill from (row, col) with new_color (4-connected)."""
    from collections import deque
    arr = grid.pixels.copy()
    target = int(arr[row, col])
    if target == new_color:
        return ArcGrid(pixels=arr, background=grid.background)
    queue: deque = deque([(row, col)])
    while queue:
        r, c = queue.popleft()
        if not (0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]):
            continue
        if arr[r, c] != target:
            continue
        arr[r, c] = new_color
        queue.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
    return ArcGrid(pixels=arr, background=grid.background)


@_register("fill_holes")
def fill_holes(grid: ArcGrid, fill_color: Optional[int] = None) -> ArcGrid:
    """Fill enclosed background holes within objects."""
    from src.core.objects.detector import ObjectDetector
    detector = ObjectDetector()
    objects = detector.detect(grid)
    arr = grid.pixels.copy()
    fc = fill_color if fill_color is not None else grid.background
    for obj in objects:
        for hole in obj.holes:
            for r, c in hole:
                arr[r, c] = fc if fill_color is not None else obj.color
    return ArcGrid(pixels=arr, background=grid.background)


# ------------------------------------------------------------------
# Border & Outline
# ------------------------------------------------------------------

@_register("add_border")
def add_border(grid: ArcGrid, color: int, thickness: int = 1) -> ArcGrid:
    """Add a uniform border of given colour and thickness around the grid."""
    H, W = grid.height, grid.width
    t = thickness
    arr = np.full((H + 2 * t, W + 2 * t), color, dtype=np.int8)
    arr[t:t + H, t:t + W] = grid.pixels
    return ArcGrid(pixels=arr, background=grid.background)


@_register("outline")
def outline(grid: ArcGrid, color: int) -> ArcGrid:
    """Draw an outline (1-px border) around all non-background objects."""
    arr = grid.pixels.copy()
    H, W = arr.shape
    new_arr = arr.copy()
    for r in range(H):
        for c in range(W):
            if arr[r, c] == grid.background:
                # Check if any neighbour is non-background
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and arr[nr, nc] != grid.background:
                        new_arr[r, c] = color
                        break
    return ArcGrid(pixels=new_arr, background=grid.background)


@_register("hollow")
def hollow(grid: ArcGrid) -> ArcGrid:
    """Keep only the border pixels of each object (make them hollow)."""
    arr = grid.pixels.copy()
    H, W = arr.shape
    new_arr = np.full_like(arr, grid.background)
    for r in range(H):
        for c in range(W):
            if arr[r, c] == grid.background:
                continue
            # Keep if it has at least one background neighbour
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W) or arr[nr, nc] == grid.background:
                    new_arr[r, c] = arr[r, c]
                    break
    return ArcGrid(pixels=new_arr, background=grid.background)


# ------------------------------------------------------------------
# Object-level Operations
# ------------------------------------------------------------------

@_register("delete_object")
def delete_object(grid: ArcGrid, color: int) -> ArcGrid:
    """Delete all pixels of a given colour (replace with background)."""
    return replace_color(grid, color, grid.background)


@_register("keep_object")
def keep_object(grid: ArcGrid, color: int) -> ArcGrid:
    """Keep only pixels of the given colour; background everything else."""
    arr = np.where(grid.pixels == color, grid.pixels, grid.background).astype(np.int8)
    return ArcGrid(pixels=arr, background=grid.background)


@_register("duplicate_object")
def duplicate_object(grid: ArcGrid, color: int, dr: int, dc: int) -> ArcGrid:
    """Duplicate the object of *color* shifted by (dr, dc)."""
    arr = grid.pixels.copy()
    positions = list(zip(*np.where(arr == color)))
    for r, c in positions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < arr.shape[0] and 0 <= nc < arr.shape[1]:
            arr[nr, nc] = color
    return ArcGrid(pixels=arr, background=grid.background)


@_register("move_object")
def move_object(grid: ArcGrid, color: int, dr: int, dc: int) -> ArcGrid:
    """Move the object of *color* by (dr, dc), erasing original position."""
    arr = grid.pixels.copy()
    positions = list(zip(*np.where(arr == color)))
    for r, c in positions:
        arr[r, c] = grid.background
    for r, c in positions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < arr.shape[0] and 0 <= nc < arr.shape[1]:
            arr[nr, nc] = color
    return ArcGrid(pixels=arr, background=grid.background)


@_register("extract_object")
def extract_object(grid: ArcGrid, color: int) -> ArcGrid:
    """Return a grid cropped to the bounding box of the *color* object."""
    positions = grid.pixels_of_color(color)
    if not positions:
        return grid.copy()
    rows = [r for r, _ in positions]
    cols = [c for _, c in positions]
    return crop(grid, min(rows), max(rows), min(cols), max(cols))


@_register("expand_object")
def expand_object(grid: ArcGrid, color: int, radius: int = 1) -> ArcGrid:
    """Dilate the object of *color* by *radius* pixels (4-connected)."""
    from collections import deque
    arr = grid.pixels.copy()
    seeds = list(zip(*np.where(arr == color)))
    front = deque(seeds)
    visited = set(seeds)
    for _ in range(radius):
        next_front: deque = deque()
        while front:
            r, c = front.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < arr.shape[0]
                    and 0 <= nc < arr.shape[1]
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    arr[nr, nc] = color
                    next_front.append((nr, nc))
        front = next_front
    return ArcGrid(pixels=arr, background=grid.background)


@_register("shrink_object")
def shrink_object(grid: ArcGrid, color: int, radius: int = 1) -> ArcGrid:
    """Erode the object of *color* by *radius* pixels (4-connected)."""
    arr = grid.pixels.copy()
    H, W = arr.shape
    for _ in range(radius):
        to_remove = []
        for r in range(H):
            for c in range(W):
                if arr[r, c] != color:
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W) or arr[nr, nc] != color:
                        to_remove.append((r, c))
                        break
        for r, c in to_remove:
            arr[r, c] = grid.background
    return ArcGrid(pixels=arr, background=grid.background)


# ------------------------------------------------------------------
# Grid Composition
# ------------------------------------------------------------------

@_register("union_grids")
def union_grids(grid_a: ArcGrid, grid_b: ArcGrid, priority: str = "a") -> ArcGrid:
    """Overlay two grids of the same size; non-background pixels take priority."""
    assert grid_a.size == grid_b.size, "Grids must have identical dimensions"
    arr = grid_a.pixels.copy()
    bg = grid_a.background
    mask_b_non_bg = grid_b.pixels != grid_b.background
    if priority == "b":
        arr[mask_b_non_bg] = grid_b.pixels[mask_b_non_bg]
    else:
        # a takes priority: only fill background cells with b
        arr[arr == bg] = grid_b.pixels[arr == bg]
    return ArcGrid(pixels=arr, background=bg)


@_register("intersect_grids")
def intersect_grids(grid_a: ArcGrid, grid_b: ArcGrid) -> ArcGrid:
    """Keep only pixels present (non-background) in BOTH grids."""
    assert grid_a.size == grid_b.size
    arr = np.where(
        (grid_a.pixels != grid_a.background) & (grid_b.pixels != grid_b.background),
        grid_a.pixels,
        grid_a.background,
    ).astype(np.int8)
    return ArcGrid(pixels=arr, background=grid_a.background)


@_register("diff_grids")
def diff_grids(grid_a: ArcGrid, grid_b: ArcGrid) -> ArcGrid:
    """Pixels in A but not in B (set difference)."""
    assert grid_a.size == grid_b.size
    arr = np.where(
        (grid_a.pixels != grid_a.background) & (grid_b.pixels == grid_b.background),
        grid_a.pixels,
        grid_a.background,
    ).astype(np.int8)
    return ArcGrid(pixels=arr, background=grid_a.background)


@_register("mask")
def mask(grid: ArcGrid, mask_grid: ArcGrid, mask_color: int) -> ArcGrid:
    """Zero out (background) pixels where mask_grid == mask_color."""
    assert grid.size == mask_grid.size
    arr = grid.pixels.copy()
    arr[mask_grid.pixels == mask_color] = grid.background
    return ArcGrid(pixels=arr, background=grid.background)


# ------------------------------------------------------------------
# Structural / Spatial Operations
# ------------------------------------------------------------------

@_register("gravity")
def gravity(grid: ArcGrid, direction: str = "down") -> ArcGrid:
    """Apply gravity: non-background pixels fall in *direction*."""
    arr = grid.pixels.copy()
    H, W = arr.shape
    bg = grid.background

    if direction == "down":
        for c in range(W):
            col = [arr[r, c] for r in range(H) if arr[r, c] != bg]
            col = [bg] * (H - len(col)) + col
            for r, v in enumerate(col):
                arr[r, c] = v
    elif direction == "up":
        for c in range(W):
            col = [arr[r, c] for r in range(H) if arr[r, c] != bg]
            col = col + [bg] * (H - len(col))
            for r, v in enumerate(col):
                arr[r, c] = v
    elif direction == "right":
        for r in range(H):
            row = [arr[r, c] for c in range(W) if arr[r, c] != bg]
            row = [bg] * (W - len(row)) + row
            for c, v in enumerate(row):
                arr[r, c] = v
    elif direction == "left":
        for r in range(H):
            row = [arr[r, c] for c in range(W) if arr[r, c] != bg]
            row = row + [bg] * (W - len(row))
            for c, v in enumerate(row):
                arr[r, c] = v
    else:
        raise ValueError(f"Unknown direction: {direction!r}")

    return ArcGrid(pixels=arr, background=bg)


@_register("sort_objects")
def sort_objects(grid: ArcGrid, key: str = "area", reverse: bool = True) -> ArcGrid:
    """Sort detected objects by *key* and re-draw them top-to-bottom."""
    from src.core.objects.detector import ObjectDetector
    detector = ObjectDetector()
    objects = detector.detect(grid)
    key_fn = {
        "area": lambda o: o.area,
        "color": lambda o: o.color,
        "x": lambda o: o.bounding_box.col_min,
        "y": lambda o: o.bounding_box.row_min,
    }.get(key, lambda o: o.area)
    objects.sort(key=key_fn, reverse=reverse)

    arr = np.full_like(grid.pixels, grid.background)
    cur_row = 0
    for obj in objects:
        snippet = obj.to_grid_snippet()
        h, w = snippet.shape
        if cur_row + h > arr.shape[0]:
            break
        arr[cur_row:cur_row + h, :w] = snippet
        cur_row += h
    return ArcGrid(pixels=arr, background=grid.background)


@_register("align_objects")
def align_objects(grid: ArcGrid, alignment: str = "left") -> ArcGrid:
    """Align all detected objects to a common edge."""
    from src.core.objects.detector import ObjectDetector
    detector = ObjectDetector()
    objects = detector.detect(grid)
    arr = np.full_like(grid.pixels, grid.background)
    H, W = arr.shape
    for obj in objects:
        snippet = obj.to_grid_snippet()
        h, w = snippet.shape
        r0 = obj.bounding_box.row_min
        if alignment == "left":
            arr[r0:r0 + h, :w] = snippet
        elif alignment == "right":
            arr[r0:r0 + h, W - w:W] = snippet
        elif alignment == "top":
            arr[:h, obj.bounding_box.col_min:obj.bounding_box.col_min + w] = snippet
        elif alignment == "bottom":
            arr[H - h:H, obj.bounding_box.col_min:obj.bounding_box.col_min + w] = snippet
    return ArcGrid(pixels=arr, background=grid.background)


@_register("reflect")
def reflect(grid: ArcGrid, axis: str = "vertical") -> ArcGrid:
    """Reflect and concatenate the grid along an axis to create symmetry."""
    if axis == "vertical":
        mirrored = mirror_horizontal(grid)
        arr = np.concatenate([grid.pixels, mirrored.pixels], axis=1)
    elif axis == "horizontal":
        mirrored = mirror_vertical(grid)
        arr = np.concatenate([grid.pixels, mirrored.pixels], axis=0)
    else:
        raise ValueError(f"Unknown axis: {axis!r}")
    return ArcGrid(pixels=arr, background=grid.background)


@_register("connect_objects")
def connect_objects(grid: ArcGrid, color_a: int, color_b: int, line_color: int) -> ArcGrid:
    """Draw a straight line between the centroids of two colour objects."""
    pos_a = grid.pixels_of_color(color_a)
    pos_b = grid.pixels_of_color(color_b)
    if not pos_a or not pos_b:
        return grid.copy()

    ar = sum(r for r, _ in pos_a) / len(pos_a)
    ac = sum(c for _, c in pos_a) / len(pos_a)
    br = sum(r for r, _ in pos_b) / len(pos_b)
    bc = sum(c for _, c in pos_b) / len(pos_b)

    arr = grid.pixels.copy()
    steps = max(abs(int(br) - int(ar)), abs(int(bc) - int(ac)), 1)
    for i in range(steps + 1):
        t = i / steps
        r = round(ar + t * (br - ar))
        c = round(ac + t * (bc - ac))
        if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
            arr[r, c] = line_color
    return ArcGrid(pixels=arr, background=grid.background)


@_register("split")
def split(grid: ArcGrid, axis: str = "horizontal", parts: int = 2) -> List[ArcGrid]:
    """Split grid into equal parts along axis. Returns list of ArcGrids."""
    arr = grid.pixels
    if axis == "horizontal":
        splits = np.array_split(arr, parts, axis=0)
    elif axis == "vertical":
        splits = np.array_split(arr, parts, axis=1)
    else:
        raise ValueError(f"Unknown axis: {axis!r}")
    return [ArcGrid(pixels=s.copy(), background=grid.background) for s in splits]


@_register("merge_grids")
def merge_grids(grids: List[ArcGrid], axis: str = "horizontal") -> ArcGrid:
    """Concatenate a list of grids along an axis."""
    arrs = [g.pixels for g in grids]
    if axis == "horizontal":
        arr = np.concatenate(arrs, axis=0)
    elif axis == "vertical":
        arr = np.concatenate(arrs, axis=1)
    else:
        raise ValueError(f"Unknown axis: {axis!r}")
    return ArcGrid(pixels=arr, background=grids[0].background)


@_register("compose")
def compose(grid: ArcGrid, operations: List[Dict]) -> ArcGrid:
    """Apply a sequence of operations to a grid (sequential composition).

    Args:
        grid:       Input grid.
        operations: List of {"name": str, "args": dict} dicts.

    Returns:
        Resulting grid after all operations.
    """
    result = grid
    for op in operations:
        name = op["name"]
        args = op.get("args", {})
        if name not in PRIMITIVE_REGISTRY:
            raise ValueError(f"Unknown primitive: {name!r}")
        result = PRIMITIVE_REGISTRY[name](result, **args)
    return result


@_register("invert_colors")
def invert_colors(grid: ArcGrid) -> ArcGrid:
    """Invert colors: 0 -> 9, 1 -> 8, ..., 9 -> 0."""
    out = grid.copy()
    out.pixels = (9 - grid.pixels).astype(np.int8)
    return out


@_register("recolor_non_background")
def recolor_non_background(grid: ArcGrid, new_color: int = 1) -> ArcGrid:
    """Replace all non-background pixels with new_color."""
    out = grid.copy()
    mask = grid.pixels != grid.background
    out.pixels[mask] = new_color
    return out


@_register("isolate_largest")
def isolate_largest(grid: ArcGrid) -> ArcGrid:
    """Keep only the largest object in the grid, backgrounding all else."""
    detector = ObjectDetector()
    objects = detector.detect(grid)
    if not objects:
        return grid.copy()
    largest = max(objects, key=lambda o: o.area)
    out = ArcGrid.empty(grid.height, grid.width, fill=grid.background)
    for r, c in largest.pixels:
        out.set(r, c, largest.color)
    return out


@_register("isolate_smallest")
def isolate_smallest(grid: ArcGrid) -> ArcGrid:
    """Keep only the smallest object in the grid, backgrounding all else."""
    detector = ObjectDetector()
    objects = detector.detect(grid)
    if not objects:
        return grid.copy()
    smallest = min(objects, key=lambda o: o.area)
    out = ArcGrid.empty(grid.height, grid.width, fill=grid.background)
    for r, c in smallest.pixels:
        out.set(r, c, smallest.color)
    return out


@_register("center_content")
def center_content(grid: ArcGrid) -> ArcGrid:
    """Center non-background content in the grid canvas."""
    non_bg = grid.pixels != grid.background
    if not non_bg.any():
        return grid.copy()
    rows, cols = np.where(non_bg)
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    ch, cw = rmax - rmin + 1, cmax - cmin + 1

    content = grid.pixels[rmin:rmax + 1, cmin:cmax + 1]
    out = ArcGrid.empty(grid.height, grid.width, fill=grid.background)

    start_r = (grid.height - ch) // 2
    start_c = (grid.width - cw) // 2
    out.pixels[start_r:start_r + ch, start_c:start_c + cw] = content
    return out


__all__ = ["PRIMITIVE_REGISTRY"] + list(PRIMITIVE_REGISTRY.keys())

# --- End src/dsl/primitives/transforms.py ---

# --- Begin src/dsl/primitives/advanced_primitives.py ---
"""
Advanced DSL Primitives — Grid morphology, pattern tiling, cellular automata,
enclosed hole filling, and outline extraction.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion



@_register("dilate")
def dilate(grid: ArcGrid, color: Optional[int] = None, radius: int = 1) -> ArcGrid:
    """
    Morphological dilation on non-background pixels (or specific color).
    Expands object boundaries by *radius* pixels.
    """
    arr = grid.pixels.copy()
    bg = grid.background

    if color is not None:
        mask = (arr == color)
        target_color = color
    else:
        mask = (arr != bg)
        # Find dominant non-background color
        non_bg_colors = arr[arr != bg]
        target_color = int(non_bg_colors[0]) if len(non_bg_colors) > 0 else 1

    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    dilated_mask = binary_dilation(mask, structure=struct)

    res = arr.copy()
    res[dilated_mask & (arr == bg)] = target_color
    return ArcGrid(pixels=res, background=bg)


@_register("erode")
def erode(grid: ArcGrid, color: Optional[int] = None, radius: int = 1) -> ArcGrid:
    """
    Morphological erosion on non-background pixels (or specific color).
    Shrinks object boundaries by *radius* pixels.
    """
    arr = grid.pixels.copy()
    bg = grid.background

    if color is not None:
        mask = (arr == color)
    else:
        mask = (arr != bg)

    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    eroded_mask = binary_erosion(mask, structure=struct)

    res = arr.copy()
    res[mask & ~eroded_mask] = bg
    return ArcGrid(pixels=res, background=bg)


@_register("extract_outline")
def extract_outline(grid: ArcGrid, color: Optional[int] = None, outline_color: Optional[int] = None) -> ArcGrid:
    """
    Extracts the outer boundary / outline of objects, leaving inner core as background.
    """
    arr = grid.pixels.copy()
    bg = grid.background
    out_col = outline_color if outline_color is not None else 1

    if color is not None:
        mask = (arr == color)
        out_col = outline_color if outline_color is not None else color
    else:
        mask = (arr != bg)

    # Erosion gives interior
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    interior = binary_erosion(mask, structure=struct)
    outline_mask = mask & ~interior

    res = np.full_like(arr, bg)
    res[outline_mask] = out_col
    return ArcGrid(pixels=res, background=bg)


@_register("fill_enclosed_holes")
def fill_enclosed_holes(grid: ArcGrid, fill_color: int = 1) -> ArcGrid:
    """
    Fills enclosed background regions (holes surrounded by non-background pixels)
    with *fill_color*.
    """
    arr = grid.pixels.copy()
    bg = grid.background
    h, w = arr.shape

    # Flood fill exterior background from grid borders
    from collections import deque
    exterior_bg = np.zeros((h, w), dtype=bool)
    queue = deque()

    for r in range(h):
        for c in (0, w - 1):
            if arr[r, c] == bg and not exterior_bg[r, c]:
                exterior_bg[r, c] = True
                queue.append((r, c))

    for c in range(w):
        for r in (0, h - 1):
            if arr[r, c] == bg and not exterior_bg[r, c]:
                exterior_bg[r, c] = True
                queue.append((r, c))

    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    while queue:
        r, c = queue.popleft()
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]
            if 0 <= nr < h and 0 <= nc < w:
                if arr[nr, nc] == bg and not exterior_bg[nr, nc]:
                    exterior_bg[nr, nc] = True
                    queue.append((nr, nc))

    # Holes are background pixels that are not exterior
    holes_mask = (arr == bg) & ~exterior_bg
    res = arr.copy()
    res[holes_mask] = fill_color
    return ArcGrid(pixels=res, background=bg)


@_register("pattern_repeat")
def pattern_repeat(grid: ArcGrid, tile_h: int = 2, tile_w: int = 2) -> ArcGrid:
    """
    Repeats the grid content as a periodic tile to fill a larger canvas.
    """
    arr = grid.pixels.copy()
    tiled = np.tile(arr, (tile_h, tile_w))
    return ArcGrid(pixels=tiled, background=grid.background)


@_register("step_ca_majority")
def step_ca_majority(grid: ArcGrid) -> ArcGrid:
    """
    Executes a 1-step majority-vote cellular automaton over non-background neighbors.
    Useful for smoothing or noise-reduction tasks.
    """
    arr = grid.pixels.copy()
    bg = grid.background
    h, w = arr.shape
    res = arr.copy()

    for r in range(h):
        for c in range(w):
            neighbors = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbors.append(arr[nr, nc])
            if neighbors:
                counts = {}
                for val in neighbors:
                    counts[val] = counts.get(val, 0) + 1
                most_common = max(counts.items(), key=lambda x: x[1])[0]
                if counts[most_common] >= 5:
                    res[r, c] = most_common

    return ArcGrid(pixels=res, background=bg)

# --- End src/dsl/primitives/advanced_primitives.py ---

# --- Begin src/dsl/primitives/higher_order.py ---
"""
Higher-Order AST Program Primitives — Object-level mapping, conditional transformations,
and line reductions.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import numpy as np



@_register("map_objects")
def map_objects(grid: ArcGrid, primitive_name: str = "rotate_90") -> ArcGrid:
    """
    Detects objects and applies specified primitive function independently to each object.
    """
    if primitive_name not in PRIMITIVE_REGISTRY:
        return grid

    primitive_fn = PRIMITIVE_REGISTRY[primitive_name]
    detector = ObjectDetector()
    objects = detector.detect(grid)

    if not objects:
        return grid

    res_arr = np.full_like(grid.pixels, grid.background)

    for obj in objects:
        bbox = obj.bounding_box
        sub_arr = grid.pixels[bbox.row_min:bbox.row_max + 1, bbox.col_min:bbox.col_max + 1].copy()
        mask = np.zeros_like(sub_arr, dtype=bool)
        for r, c in obj.pixels:
            mask[r - bbox.row_min, c - bbox.col_min] = True

        # Mask background
        sub_arr[~mask] = grid.background
        sub_grid = ArcGrid(pixels=sub_arr, background=grid.background)

        try:
            trans_grid = primitive_fn(sub_grid)
            th, tw = trans_grid.pixels.shape
            r_end = min(bbox.row_min + th, grid.height)
            c_end = min(bbox.col_min + tw, grid.width)
            sub_trans = trans_grid.pixels[:r_end - bbox.row_min, :c_end - bbox.col_min]

            t_mask = (sub_trans != grid.background)
            res_arr[bbox.row_min:r_end, bbox.col_min:c_end][t_mask] = sub_trans[t_mask]
        except Exception:
            for r, c in obj.pixels:
                res_arr[r, c] = obj.color

    return ArcGrid(pixels=res_arr, background=grid.background)


@_register("conditional_apply")
def conditional_apply(
    grid: ArcGrid,
    condition: str = "is_symmetric",
    true_primitive: str = "mirror_horizontal",
    false_primitive: str = "rotate_90",
) -> ArcGrid:
    """
    Applies true_primitive if grid satisfies condition (e.g. is_symmetric, has_holes),
    else applies false_primitive.
    """
    arr = grid.pixels.copy()
    bg = grid.background

    is_true = False
    if condition == "is_symmetric":
        is_true = (arr == np.fliplr(arr)).all() or (arr == np.flipud(arr)).all()
    elif condition == "has_holes":
        from src.dsl.primitives.advanced_primitives import fill_enclosed_holes
        filled = fill_enclosed_holes(grid)
        is_true = (filled.pixels != arr).any()

    target_primitive = true_primitive if is_true else false_primitive
    if target_primitive in PRIMITIVE_REGISTRY:
        return PRIMITIVE_REGISTRY[target_primitive](grid)

    return grid


@_register("reduce_lines")
def reduce_lines(grid: ArcGrid, axis: int = 0, mode: str = "or") -> ArcGrid:
    """
    Reduces grid along axis (0 for rows, 1 for columns) using logical/bitwise combination.
    """
    arr = grid.pixels.copy()
    bg = grid.background

    if axis == 0:
        # Collapse rows -> 1 x W
        res = np.zeros((1, arr.shape[1]), dtype=int)
        for c in range(arr.shape[1]):
            col = arr[:, c]
            non_bg = col[col != bg]
            res[0, c] = non_bg[0] if len(non_bg) > 0 else bg
    else:
        # Collapse columns -> H x 1
        res = np.zeros((arr.shape[0], 1), dtype=int)
        for r in range(arr.shape[0]):
            row = arr[r, :]
            non_bg = row[row != bg]
            res[r, 0] = non_bg[0] if len(non_bg) > 0 else bg

    return ArcGrid(pixels=res, background=bg)

# --- End src/dsl/primitives/higher_order.py ---

# --- Begin src/dsl/executor/executor.py ---
"""
DSL Executor — compile and run DSLProgram instances on ArcGrid objects.

Features:
  - Deterministic execution with full debug trace
  - Condition evaluation at runtime
  - Timeout / step budget enforcement
  - Sandboxed: never modifies input grids in-place
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

    Condition,
    ConditionType,
    DSLInstruction,
    DSLProgram,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Execution trace
# ---------------------------------------------------------------------------

@dataclass
class StepTrace:
    """Record of a single instruction execution."""

    step_index: int
    primitive: str
    args: Dict[str, Any]
    condition_result: bool
    skipped: bool
    success: bool
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    input_shape: tuple = ()
    output_shape: tuple = ()


@dataclass
class ExecutionTrace:
    """Complete trace of a program execution."""

    program_name: str
    steps: List[StepTrace] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    total_elapsed_ms: float = 0.0

    def summary(self) -> str:
        executed = sum(1 for s in self.steps if not s.skipped and s.success)
        skipped = sum(1 for s in self.steps if s.skipped)
        errors = sum(1 for s in self.steps if s.error)
        return (
            f"Program '{self.program_name}': "
            f"{executed} executed, {skipped} skipped, {errors} errors, "
            f"{self.total_elapsed_ms:.1f}ms total"
        )


# ---------------------------------------------------------------------------
# Condition evaluator
# ---------------------------------------------------------------------------

class ConditionEvaluator:
    """Evaluate Condition predicates against the current grid state."""

    def __init__(self, detector: Optional[ObjectDetector] = None) -> None:
        self._detector = detector or ObjectDetector()

    def evaluate(self, condition: Condition, grid: ArcGrid) -> bool:
        ct = condition.ctype
        p = condition.params

        if ct == ConditionType.ALWAYS:
            return True
        if ct == ConditionType.NEVER:
            return False

        if ct == ConditionType.COLOR_COUNT:
            color = p["color"]
            op = p["op"]        # "eq", "lt", "gt", "le", "ge"
            threshold = p["threshold"]
            count = int((grid.pixels == color).sum())
            return {
                "eq": count == threshold,
                "lt": count < threshold,
                "gt": count > threshold,
                "le": count <= threshold,
                "ge": count >= threshold,
            }[op]

        if ct == ConditionType.OBJECT_HAS_TAG:
            tag = p["tag"]
            objects = self._detector.detect(grid)
            return any(tag in obj.tags for obj in objects)

        if ct == ConditionType.GRID_SYMMETRIC:
            axis = p.get("axis", "horizontal")
            import numpy as np
            if axis == "horizontal":
                return bool(np.array_equal(grid.pixels, grid.pixels[:, ::-1]))
            return bool(np.array_equal(grid.pixels, grid.pixels[::-1, :]))

        logger.warning(f"Unknown condition type: {ct}")
        return True


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class DSLExecutor:
    """Execute DSLProgram instances on ArcGrid inputs.

    Usage::

        executor = DSLExecutor()
        result, trace = executor.execute(program, input_grid)
    """

    def __init__(
        self,
        max_steps: int = 100,
        timeout_sec: float = 5.0,
        debug: bool = False,
    ) -> None:
        """
        Args:
            max_steps:   Maximum instructions to execute (safety limit).
            timeout_sec: Wall-clock timeout in seconds.
            debug:       If True, log each step.
        """
        self.max_steps = max_steps
        self.timeout_sec = timeout_sec
        self.debug = debug
        self._cond_eval = ConditionEvaluator()

    def execute(
        self,
        program: DSLProgram,
        grid: ArcGrid,
    ) -> tuple[Optional[ArcGrid], ExecutionTrace]:
        """Execute a DSL program on a grid.

        Args:
            program: The DSLProgram to execute.
            grid:    Input ArcGrid (not mutated).

        Returns:
            Tuple of (result_grid_or_None, ExecutionTrace).
        """
        trace = ExecutionTrace(program_name=program.name or repr(program))
        current = grid.copy()
        t_start = time.perf_counter()

        try:
            for step_idx, instruction in enumerate(program.instructions):
                if step_idx >= self.max_steps:
                    trace.error = f"Exceeded max_steps={self.max_steps}"
                    trace.success = False
                    break

                elapsed = (time.perf_counter() - t_start) * 1000
                if elapsed / 1000 > self.timeout_sec:
                    trace.error = f"Timeout after {elapsed:.0f}ms"
                    trace.success = False
                    break

                step_trace = self._execute_step(step_idx, instruction, current)
                trace.steps.append(step_trace)

                if step_trace.success and not step_trace.skipped:
                    # The primitive may return a new grid — apply it
                    # (some primitives return lists, handled separately)
                    new_grid = self._call_primitive(instruction, current)
                    if isinstance(new_grid, ArcGrid):
                        current = new_grid

                elif not step_trace.success:
                    trace.success = False
                    trace.error = step_trace.error
                    break

        except Exception as e:
            trace.success = False
            trace.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"Executor crashed: {trace.error}")
            return None, trace

        trace.total_elapsed_ms = (time.perf_counter() - t_start) * 1000
        if self.debug:
            logger.debug(trace.summary())

        return current if trace.success else None, trace

    def execute_on_pairs(
        self,
        program: DSLProgram,
        pairs: List[Dict],
    ) -> List[tuple[Optional[ArcGrid], ExecutionTrace]]:
        """Execute a program on multiple (input, expected_output) pairs.

        Args:
            program: The DSLProgram.
            pairs:   List of dicts with "input" key (ArcGrid).

        Returns:
            List of (result, trace) per pair.
        """
        return [self.execute(program, pair["input"]) for pair in pairs]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _execute_step(
        self,
        step_idx: int,
        instruction: DSLInstruction,
        grid: ArcGrid,
    ) -> StepTrace:
        """Execute a single instruction step."""
        t0 = time.perf_counter()

        # Evaluate condition
        try:
            cond_result = self._cond_eval.evaluate(instruction.condition, grid)
        except Exception as e:
            return StepTrace(
                step_index=step_idx,
                primitive=instruction.primitive,
                args=instruction.args,
                condition_result=False,
                skipped=True,
                success=True,
                error=f"Condition error: {e}",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        if not cond_result:
            return StepTrace(
                step_index=step_idx,
                primitive=instruction.primitive,
                args=instruction.args,
                condition_result=False,
                skipped=True,
                success=True,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        # Execute primitive
        try:
            result = self._call_primitive(instruction, grid)
            success = True
            error = None
            out_shape = result.size if isinstance(result, ArcGrid) else ()
        except Exception as e:
            success = False
            error = f"{type(e).__name__}: {e}"
            out_shape = ()
            if self.debug:
                logger.debug(f"Step {step_idx} '{instruction.primitive}' failed: {error}")

        elapsed = (time.perf_counter() - t0) * 1000
        return StepTrace(
            step_index=step_idx,
            primitive=instruction.primitive,
            args=instruction.args,
            condition_result=cond_result,
            skipped=False,
            success=success,
            error=error,
            elapsed_ms=elapsed,
            input_shape=grid.size,
            output_shape=out_shape,
        )

    def _call_primitive(self, instruction: DSLInstruction, grid: ArcGrid) -> Any:
        """Look up and call the registered primitive."""
        fn = PRIMITIVE_REGISTRY.get(instruction.primitive)
        if fn is None:
            raise ValueError(f"Unknown primitive: '{instruction.primitive}'")
        return fn(grid, **instruction.args)


__all__ = ["DSLExecutor", "ExecutionTrace", "StepTrace"]

# --- End src/dsl/executor/executor.py ---

# --- Begin src/search/neural_heuristic.py ---
"""
Neural/LLM-Guided Search & Multi-Dimensional Heuristic — Advanced continuous similarity
and primitive prior probability scoring for MCTS / Beam search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np



@dataclass
class HeuristicBreakdown:
    total_score: float
    iou_score: float
    entropy_similarity: float
    color_distribution_score: float
    dimension_score: float


class MultiDimensionalHeuristic:
    """Advanced composite similarity evaluator using IoU, structural entropy, and color alignment."""

    def __init__(self) -> None:
        self.detector = ObjectDetector()

    def _compute_entropy(self, grid: ArcGrid) -> float:
        """Computes normalized Shannon entropy of pixel color distribution."""
        pixels = grid.pixels.flatten()
        if len(pixels) == 0:
            return 0.0
        _, counts = np.unique(pixels, return_counts=True)
        probs = counts / len(pixels)
        ent = -np.sum(probs * np.log2(probs + 1e-9))
        max_ent = np.log2(10.0)
        return float(ent / max_ent)

    def _compute_iou(self, predicted: ArcGrid, expected: ArcGrid) -> float:
        """Computes non-background pixel Intersection-over-Union (IoU)."""
        if (predicted.height, predicted.width) != (expected.height, expected.width):
            return 0.0
        p_mask = (predicted.pixels != predicted.background)
        e_mask = (expected.pixels != expected.background)

        intersection = np.logical_and(p_mask, e_mask).sum()
        union = np.logical_or(p_mask, e_mask).sum()

        if union == 0:
            return 1.0 if (predicted.pixels == expected.pixels).all() else 0.0
        return float(intersection / union)

    def evaluate(self, predicted: ArcGrid, expected: ArcGrid) -> HeuristicBreakdown:
        if predicted == expected:
            return HeuristicBreakdown(
                total_score=1.0,
                iou_score=1.0,
                entropy_similarity=1.0,
                color_distribution_score=1.0,
                dimension_score=1.0,
            )

        # Dimension score
        h_ratio = min(predicted.height, expected.height) / max(predicted.height, expected.height)
        w_ratio = min(predicted.width, expected.width) / max(predicted.width, expected.width)
        dim_score = (h_ratio + w_ratio) / 2.0

        # Entropy score
        p_ent = self._compute_entropy(predicted)
        e_ent = self._compute_entropy(expected)
        ent_score = 1.0 - abs(p_ent - e_ent)

        # Color distribution score
        p_hist = np.bincount(predicted.pixels.flatten(), minlength=10)
        e_hist = np.bincount(expected.pixels.flatten(), minlength=10)
        p_norm = p_hist / (p_hist.sum() + 1e-9)
        e_norm = e_hist / (e_hist.sum() + 1e-9)
        color_score = float(np.sum(np.minimum(p_norm, e_norm)))

        # IoU score
        iou = self._compute_iou(predicted, expected)

        # Composite weighted sum
        total = 0.4 * iou + 0.3 * color_score + 0.15 * ent_score + 0.15 * dim_score
        return HeuristicBreakdown(
            total_score=float(total),
            iou_score=float(iou),
            entropy_similarity=float(ent_score),
            color_distribution_score=float(color_score),
            dimension_score=float(dim_score),
        )


class NeuralSearchPrior:
    """Predicts prior probability distributions over primitive operations given input/output pairs."""

    def __init__(self) -> None:
        self.primitive_names = sorted(list(PRIMITIVE_REGISTRY.keys()))

    def predict_priors(self, input_grid: ArcGrid, target_grid: ArcGrid) -> Dict[str, float]:
        """Returns normalized prior probabilities over available primitive operations."""
        priors: Dict[str, float] = {name: 1.0 for name in self.primitive_names}

        # Spatial transformation heuristics
        if (input_grid.height, input_grid.width) == (target_grid.height, target_grid.width):
            # Same shape encourages rotations, recoloring, reflection
            for p in ["rotate_90", "rotate_180", "mirror_horizontal", "mirror_vertical", "replace_color"]:
                if p in priors:
                    priors[p] += 2.0

        elif input_grid.height * 2 == target_grid.height or input_grid.width * 2 == target_grid.width:
            # Scale heuristics
            for p in ["scale_2x", "pattern_repeat"]:
                if p in priors:
                    priors[p] += 3.0

        # Normalize to probability distribution
        total = sum(priors.values())
        return {k: v / total for k, v in priors.items()}

# --- End src/search/neural_heuristic.py ---

# --- Begin src/submission/submission_generator.py ---
"""
Submission Generator — produce competition-ready submission.json.

ARC Prize submission format (per test input, 2 attempts):
{
  "<task_id>": [
    {
      "attempt_1": [[0,1,2], ...],
      "attempt_2": [[0,1,2], ...]
    },
    ...  (one dict per test input in the task)
  ],
  ...
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = get_logger(__name__)


class SubmissionGenerator:
    """Generate submission.json for the ARC Prize competition.

    Usage::

        gen = SubmissionGenerator()
        gen.add_task_result("007bbfb7", predictions=[[grid1, grid2], [grid3, grid4]])
        gen.save("submission.json")
    """

    def __init__(self, validate: bool = True) -> None:
        self.validate = validate
        self._submission: Dict[str, Any] = {}

    def add_task_result(
        self,
        task_id: str,
        predictions: List[List[List[List[int]]]],
    ) -> None:
        """Add predictions for one task.

        Args:
            task_id:     Task identifier (filename stem).
            predictions: For each test input: a list of up to 2 candidate grids.
                         predictions[i] = [attempt_1_grid, attempt_2_grid]
        """
        task_entry = []
        for test_idx, attempts in enumerate(predictions):
            entry: Dict[str, Any] = {}
            for attempt_num, grid in enumerate(attempts[:2], start=1):
                entry[f"attempt_{attempt_num}"] = grid
            # If fewer than 2 attempts, duplicate the first
            if len(attempts) == 1:
                entry["attempt_2"] = attempts[0]
            elif len(attempts) == 0:
                # Fallback: 1x1 grid of zeros
                entry["attempt_1"] = [[0]]
                entry["attempt_2"] = [[0]]
            task_entry.append(entry)

        self._submission[task_id] = task_entry

        if self.validate:
            errors = self._validate_task_entry(task_id, task_entry)
            if errors:
                logger.warning(f"Submission validation warnings for {task_id}: {errors}")

    def add_from_solver_result(self, solver_result: Any) -> None:
        """Convenience: add from a SolverResult object.

        Args:
            solver_result: SolverResult from ARCSolver.solve()
        """
        # Group predictions by test input
        predictions = []
        for test_result in solver_result.test_results:
            predictions.append(test_result.predictions)

        self.add_task_result(solver_result.task_id, predictions)

    def save(self, output_path: str | Path, indent: int = 2) -> None:
        """Write submission.json to disk.

        Args:
            output_path: Output file path.
            indent:      JSON indentation (0 for compact).
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.validate:
            all_errors = self._validate_submission()
            if all_errors:
                logger.warning(f"Submission has {len(all_errors)} validation issues")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self._submission,
                f,
                indent=indent if indent > 0 else None,
            )

        n_tasks = len(self._submission)
        n_tests = sum(len(v) for v in self._submission.values())
        logger.info(
            f"Submission saved: {path} "
            f"({n_tasks} tasks, {n_tests} test inputs)"
        )

    def to_dict(self) -> Dict:
        return dict(self._submission)

    def summary(self) -> str:
        n_tasks = len(self._submission)
        n_tests = sum(len(v) for v in self._submission.values())
        return f"SubmissionGenerator: {n_tasks} tasks, {n_tests} total test inputs"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_task_entry(self, task_id: str, entry: List[Dict]) -> List[str]:
        errors = []
        for i, test_dict in enumerate(entry):
            for key in ("attempt_1", "attempt_2"):
                if key not in test_dict:
                    errors.append(f"[{task_id}][{i}] missing {key}")
                    continue
                grid = test_dict[key]
                if not isinstance(grid, list) or not grid:
                    errors.append(f"[{task_id}][{i}][{key}] empty or non-list grid")
                    continue
                width = len(grid[0]) if grid else 0
                for row_idx, row in enumerate(grid):
                    if not isinstance(row, list) or len(row) != width:
                        errors.append(
                            f"[{task_id}][{i}][{key}][{row_idx}] jagged row"
                        )
                    for cell in row:
                        if not isinstance(cell, int) or not (0 <= cell <= 9):
                            errors.append(
                                f"[{task_id}][{i}][{key}] invalid cell value {cell!r}"
                            )
        return errors

    def _validate_submission(self) -> List[str]:
        errors = []
        for task_id, entry in self._submission.items():
            errors.extend(self._validate_task_entry(task_id, entry))
        return errors


__all__ = ["SubmissionGenerator"]

# --- End src/submission/submission_generator.py ---

