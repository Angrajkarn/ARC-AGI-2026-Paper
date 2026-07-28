"""
SkeletonEulerSignature — Computes topological Euler signatures on skeleton coordinate sets.
"""

from __future__ import annotations

from typing import Set, Tuple


class SkeletonEulerSignature:
    """Calculates vertex, edge count and euler characteristic on skeleton representations."""

    @staticmethod
    def get_euler_characteristic(skeleton_pixels: Set[Tuple[int, int]]) -> int:
        """Returns Euler Characteristic (V - E) on 8-connected skeletal graphs."""
        if not skeleton_pixels:
            return 0

        # Vertices (V) is the number of pixels in the skeleton
        V = len(skeleton_pixels)

        # Edges (E) represents unique adjacent pixel connections (8-connected)
        edges = set()
        for r, c in skeleton_pixels:
            # Check only forward direction neighbors to prevent double counting
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in skeleton_pixels:
                    # Sort nodes to keep edge representation unique
                    p1 = (r, c)
                    p2 = (nr, nc)
                    edge = (min(p1, p2), max(p1, p2))
                    edges.add(edge)

        E = len(edges)
        return V - E
