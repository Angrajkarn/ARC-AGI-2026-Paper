"""
SkeletonHomologyMatcher — Computes cycle homology invariants on topological skeletal graphs.
"""

from __future__ import annotations

from typing import Set, Tuple


class SkeletonHomologyMatcher:
    """Matches graph topological cycle invariants on skeletal representations."""

    @staticmethod
    def compute_cycle_rank(skeleton_pixels: Set[Tuple[int, int]]) -> int:
        """Returns cycle rank (betti-1 number) of skeletal graph: E - V + C."""
        if not skeleton_pixels:
            return 0

        # Calculate connected components (C)
        from collections import deque
        visited = set()
        components = 0

        for p in skeleton_pixels:
            if p not in visited:
                components += 1
                queue = deque([p])
                visited.add(p)
                while queue:
                    curr_r, curr_c = queue.popleft()
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            neighbor = (curr_r + dr, curr_c + dc)
                            if neighbor in skeleton_pixels and neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)

        # Calculate Vertices (V) and Edges (E)
        V = len(skeleton_pixels)
        edges = set()
        for r, c in skeleton_pixels:
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in skeleton_pixels:
                    p1 = (r, c)
                    p2 = (nr, nc)
                    edges.add((min(p1, p2), max(p1, p2)))

        E = len(edges)
        # Cycle rank formula: E - V + C
        return E - V + components
