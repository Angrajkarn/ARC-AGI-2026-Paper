"""
BettiNumberMatcher — Computes Betti numbers (connected components and enclosed loops).
"""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np


class BettiNumberMatcher:
    """Calculates topological Betti numbers on object masks."""

    @staticmethod
    def get_betti_numbers(object_pixels: Set[Tuple[int, int]]) -> Tuple[int, int]:
        """Returns Betti-0 (components count) and Betti-1 (holes count)."""
        if not object_pixels:
            return 0, 0

        # Calculate Betti-0 by finding connected components
        from collections import deque
        visited = set()
        components = 0

        for r, c in object_pixels:
            if (r, c) not in visited:
                components += 1
                # Flood fill component
                queue = deque([(r, c)])
                visited.add((r, c))
                while queue:
                    curr_r, curr_c = queue.popleft()
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = curr_r + dr, curr_c + dc
                            if (nr, nc) in object_pixels and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                queue.append((nr, nc))

        # Calculate Betti-1 using Euler characteristic on bounding subgrid
        # Betti-1 (holes) count is calculated by flood filling background holes
        rows = [r for r, c in object_pixels]
        cols = [c for r, c in object_pixels]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)

        h = max_r - min_r + 3
        w = max_c - min_c + 3

        grid = np.zeros((h, w), dtype=np.uint8)
        for r, c in object_pixels:
            grid[r - min_r + 1, c - min_c + 1] = 1

        ext_visited = np.zeros((h, w), dtype=bool)
        ext_queue = deque([(0, 0)])
        ext_visited[0, 0] = True

        while ext_queue:
            curr_r, curr_c = ext_queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if not ext_visited[nr, nc] and grid[nr, nc] == 0:
                        ext_visited[nr, nc] = True
                        ext_queue.append((nr, nc))

        holes = 0
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 0 and not ext_visited[r, c]:
                    holes += 1
                    # Flood fill the hole component to mark all its pixels
                    hole_queue = deque([(r, c)])
                    ext_visited[r, c] = True
                    while hole_queue:
                        hr, hc = hole_queue.popleft()
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nhr, nhc = hr + dr, hc + dc
                            if 0 <= nhr < h and 0 <= nhc < w:
                                if not ext_visited[nhr, nhc] and grid[nhr, nhc] == 0:
                                    ext_visited[nhr, nhc] = True
                                    hole_queue.append((nhr, nhc))

        return components, holes
