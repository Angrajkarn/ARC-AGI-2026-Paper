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

from src.core.grid.grid import ArcGrid
from src.core.objects.arc_object import ArcObject, BoundingBox


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
                    if visited[r, c]:
                        continue
                    visited[r, c] = True
                    if grid.get(r, c) != color:
                        continue
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
