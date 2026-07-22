"""
TopologyInvariantMatcher — Computes persistent homology invariants (Betti-0, Betti-1, Euler characteristic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid
from src.core.objects.detector import ObjectDetector


@dataclass
class TopologySignature:
    betti_0: int  # Number of connected components
    betti_1: int  # Number of loops / enclosed holes
    euler_characteristic: int  # Betti_0 - Betti_1


class TopologyInvariantMatcher:
    """Computes topological signatures and verifies homology conservation between grid transformations."""

    def __init__(self) -> None:
        self.detector = ObjectDetector(connectivity=4, ignore_background=True)

    def compute_signature(self, grid: ArcGrid) -> TopologySignature:
        """Computes topological signature signature for non-background pixels in grid."""
        # Betti-0: count of connected objects
        objects = self.detector.detect(grid)
        b0 = len(objects)

        # Betti-1: count of background-colored holes enclosed by any object
        b1 = 0
        h, w = grid.height, grid.width

        # Detect holes by running BFS from grid boundary to find non-enclosed background pixels
        visited: Set[Tuple[int, int]] = set()
        queue = []

        # Seed queue with all boundaries
        for r in range(h):
            if grid.get(r, 0) == grid.background:
                queue.append((r, 0))
                visited.add((r, 0))
            if grid.get(r, w - 1) == grid.background:
                queue.append((r, w - 1))
                visited.add((r, w - 1))
        for c in range(w):
            if grid.get(0, c) == grid.background:
                queue.append((0, c))
                visited.add((0, c))
            if grid.get(h - 1, c) == grid.background:
                queue.append((h - 1, c))
                visited.add((h - 1, c))

        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if (nr, nc) not in visited and grid.get(nr, nc) == grid.background:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        # Any background pixel NOT visited is enclosed (part of a hole)
        enclosed_background_pixels = set()
        for r in range(h):
            for c in range(w):
                if grid.get(r, c) == grid.background and (r, c) not in visited:
                    enclosed_background_pixels.add((r, c))

        # Count connected components of enclosed background pixels to find Betti-1
        visited_holes = set()
        for hr, hc in enclosed_background_pixels:
            if (hr, hc) not in visited_holes:
                b1 += 1
                hole_queue = [(hr, hc)]
                visited_holes.add((hr, hc))
                while hole_queue:
                    r, c = hole_queue.pop(0)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in enclosed_background_pixels and (nr, nc) not in visited_holes:
                            visited_holes.add((nr, nc))
                            hole_queue.append((nr, nc))

        return TopologySignature(
            betti_0=b0,
            betti_1=b1,
            euler_characteristic=b0 - b1,
        )

    def is_invariant(self, grid_a: ArcGrid, grid_b: ArcGrid) -> bool:
        """Checks if two grids share the exact same topological signature."""
        sig_a = self.compute_signature(grid_a)
        sig_b = self.compute_signature(grid_b)
        return sig_a == sig_b
