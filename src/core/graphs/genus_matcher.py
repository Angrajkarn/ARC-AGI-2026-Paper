"""
GenusMatcher — Computes topological Genus (number of enclosed loops/holes) in object shapes.
"""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np


class GenusMatcher:
    """Calculates topological properties on coordinate patterns."""

    @staticmethod
    def compute_genus(object_pixels: Set[Tuple[int, int]]) -> int:
        """Finds number of background holes fully enclosed inside the object's pixel envelope."""
        if not object_pixels:
            return 0

        # Build bounding grid envelope
        rows = [r for r, c in object_pixels]
        cols = [c for r, c in object_pixels]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)

        h = max_r - min_r + 3
        w = max_c - min_c + 3

        grid = np.zeros((h, w), dtype=np.uint8)
        for r, c in object_pixels:
            grid[r - min_r + 1, c - min_c + 1] = 1

        # Use flood fill from border (0,0) to identify the background connected to the exterior
        from collections import deque
        visited = np.zeros((h, w), dtype=bool)
        queue = deque([(0, 0)])
        visited[0, 0] = True

        while queue:
            curr_r, curr_c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if not visited[nr, nc] and grid[nr, nc] == 0:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

        # Holes are background coordinates (grid == 0) that were not reachable from the exterior border
        holes_count = 0
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 0 and not visited[r, c]:
                    # Flood fill the enclosed hole to count it as one connected component hole
                    holes_count += 1
                    # Flood fill hole
                    hole_queue = deque([(r, c)])
                    visited[r, c] = True
                    while hole_queue:
                        hr, hc = hole_queue.popleft()
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nhr, nhc = hr + dr, hc + dc
                            if 0 <= nhr < h and 0 <= nhc < w:
                                if not visited[nhr, nhc] and grid[nhr, nhc] == 0:
                                    visited[nhr, nhc] = True
                                    hole_queue.append((nhr, nhc))

        return holes_count
