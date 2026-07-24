"""
ContourTracer — Traces boundary perimeter coordinates of segmented object structures.
"""

from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid


class ContourTracer:
    """Finds outer boundary contours of objects."""

    @staticmethod
    def trace_contour(grid: ArcGrid, object_pixels: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Returns ordered list of perimeter coordinates matching the outer edge."""
        if not object_pixels:
            return []

        contour: List[Tuple[int, int]] = []
        h, w = grid.pixels.shape

        # Identify perimeter coordinates: coordinates inside object_pixels that have a background neighbor
        bg = grid.background
        for r, c in object_pixels:
            is_edge = False
            # Check 8-way neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if grid.pixels[nr, nc] == bg:
                            is_edge = True
                            break
                    else:
                        is_edge = True  # Grid borders are contours
                        break
                if is_edge:
                    break

            if is_edge:
                contour.append((r, c))

        # Sort contour coordinates lexicographically
        contour.sort()
        return contour
