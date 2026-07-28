"""
SkeletonJunctionAnalyzer — Detects junction nodes (3+ neighbors) and endpoints (1 neighbor) on skeletons.
"""

from __future__ import annotations

from typing import Set, Tuple


class SkeletonJunctionAnalyzer:
    """Classifies skeletal node degree types in 8-connected graph representations."""

    @staticmethod
    def get_junctions_and_endpoints(
        skeleton_pixels: Set[Tuple[int, int]]
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """Returns (junction_set, endpoint_set)."""
        junctions = set()
        endpoints = set()

        for r, c in skeleton_pixels:
            neighbors = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r + dr, c + dc) in skeleton_pixels:
                    neighbors += 1

            if neighbors >= 3:
                junctions.add((r, c))
            elif neighbors == 1:
                endpoints.add((r, c))

        return junctions, endpoints
