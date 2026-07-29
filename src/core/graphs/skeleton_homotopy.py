"""
SkeletonHomotopyExtractor — Extracts fundamental group pi_1(X) loop generators from skeletal graphs.
"""

from __future__ import annotations

from typing import List, Set, Tuple


class SkeletonHomotopyExtractor:
    """Computes fundamental group loop generator counts and cycle basis representation."""

    @staticmethod
    def extract_homotopy_generators(skeleton_pixels: Set[Tuple[int, int]]) -> int:
        """Returns rank of pi_1(X) fundamental group generators (1-cycles / independent loops)."""
        if not skeleton_pixels:
            return 0

        # Count edges in 4-neighborhood
        nodes = len(skeleton_pixels)
        edges = 0
        pixel_list = list(skeleton_pixels)
        pixel_set = set(skeleton_pixels)

        visited_edges = set()
        for r, c in pixel_list:
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in pixel_set:
                    edges += 1

        # Connected components via BFS
        components = 0
        unvisited = set(skeleton_pixels)

        while unvisited:
            components += 1
            start = unvisited.pop()
            queue = [start]
            while queue:
                curr_r, curr_c = queue.pop(0)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = (curr_r + dr, curr_c + dc)
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        queue.append(neighbor)

        # Euler characteristic formula for graph: beta_1 = E - V + C
        betti_1 = max(0, edges - nodes + components)
        return betti_1
