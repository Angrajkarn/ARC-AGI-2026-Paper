"""
SkeletonLoopPerimeterDetector — Measures boundary perimeter lengths around skeletal loops.
"""

from __future__ import annotations

from typing import Set, Tuple


class SkeletonLoopPerimeterDetector:
    """Calculates perimeter pixel distance metrics around skeletal loop structures."""

    @staticmethod
    def measure_loop_perimeter(skeleton_pixels: Set[Tuple[int, int]]) -> int:
        """Returns boundary pixel count (number of pixels with <4 orthogonal skeletal neighbors)."""
        if not skeleton_pixels:
            return 0

        perimeter = 0
        for r, c in skeleton_pixels:
            neighbors = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r + dr, c + dc) in skeleton_pixels:
                    neighbors += 1

            if neighbors < 4:
                perimeter += 1

        return perimeter
