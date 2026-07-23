"""
PathJunctionMatcher — Classifies thinned skeletal grid coordinates into endpoints, pathways, and intersection junctions.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid


class PathJunctionMatcher:
    """Identifies and classifies structural path routing coordinates on thinned grid skeletons."""

    @staticmethod
    def classify_coordinates(skeleton_grid: ArcGrid) -> Dict[str, List[Tuple[int, int]]]:
        """Classifies each non-background skeletal coordinate by orthogonal connection degree."""
        pixels = skeleton_grid.pixels
        h, w = pixels.shape
        bg = skeleton_grid.background

        endpoints: List[Tuple[int, int]] = []
        pathways: List[Tuple[int, int]] = []
        junctions: List[Tuple[int, int]] = []

        for r in range(h):
            for c in range(w):
                if pixels[r, c] != bg:
                    # Count 4-way orthogonal neighbors
                    n_count = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if pixels[nr, nc] != bg:
                                n_count += 1

                    coord = (r, c)
                    if n_count == 1:
                        endpoints.append(coord)
                    elif n_count == 2:
                        pathways.append(coord)
                    elif n_count >= 3:
                        junctions.append(coord)

        return {
            "endpoints": endpoints,
            "pathways": pathways,
            "junctions": junctions,
        }
