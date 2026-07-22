"""
TopologySkeletonizer — extracts 1-pixel wide medial axes (skeletons) from grid shapes for path-routing validation.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class TopologySkeletonizer:
    """Implements topological pixel thinning (skeletonization) on binary grid masks."""

    @staticmethod
    def extract_skeleton(grid: ArcGrid) -> ArcGrid:
        """Thins all non-background shapes into 1-pixel wide lines, preserving topology."""
        pixels = (grid.pixels != grid.background).astype(np.uint8)
        h, w = grid.height, grid.width

        # Simple iterative boundary thinning
        changed = True
        while changed:
            changed = False
            to_remove = []

            for r in range(1, h - 1):
                for c in range(1, w - 1):
                    if pixels[r, c] == 1:
                        # Count neighbors
                        neighbors = [
                            pixels[r - 1, c],
                            pixels[r - 1, c + 1],
                            pixels[r, c + 1],
                            pixels[r + 1, c + 1],
                            pixels[r + 1, c],
                            pixels[r + 1, c - 1],
                            pixels[r, c - 1],
                            pixels[r - 1, c - 1],
                        ]
                        n_count = sum(neighbors)

                        # Thin boundary points that do not break connectivity (simple heuristic count)
                        if 2 <= n_count <= 6:
                            # Verify if removing changes Euler/connectivity number
                            # (Here we approximate: check if it acts as a boundary segment)
                            transitions = 0
                            for i in range(8):
                                if neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1:
                                    transitions += 1

                            if transitions == 1:
                                to_remove.append((r, c))

            if to_remove:
                for r, c in to_remove:
                    pixels[r, c] = 0
                changed = True

        # Re-apply color representation on thinned skeleton
        skel_pixels = np.full_like(grid.pixels, grid.background)
        skel_pixels[pixels == 1] = 1

        return ArcGrid(pixels=skel_pixels, background=grid.background)
