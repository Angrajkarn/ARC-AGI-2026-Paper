"""
SkeletonConnectivityValidator — Validates connectivity status across topological skeleton segments.
"""

from __future__ import annotations

from typing import Set, Tuple


class SkeletonConnectivityValidator:
    """Checks whether topological skeleton points form a single connected component."""

    @staticmethod
    def is_connected(skeleton_pixels: Set[Tuple[int, int]]) -> bool:
        """Returns True if all skeletal points are connected via 8-connectivity."""
        if not skeleton_pixels:
            return True

        from collections import deque
        start_pixel = next(iter(skeleton_pixels))
        visited = {start_pixel}
        queue = deque([start_pixel])

        while queue:
            curr_r, curr_c = queue.popleft()
            # Check 8-neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = curr_r + dr, curr_c + dc
                    if (nr, nc) in skeleton_pixels and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        return len(visited) == len(skeleton_pixels)
