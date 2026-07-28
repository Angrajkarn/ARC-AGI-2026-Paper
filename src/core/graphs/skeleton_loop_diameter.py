"""
SkeletonLoopDiameterEstimator — Measures bounding diameter across skeletal loop enclosures.
"""

from __future__ import annotations

import math
from typing import Set, Tuple


class SkeletonLoopDiameterEstimator:
    """Calculates maximum pairwise spatial distance across skeletal loop points."""

    @staticmethod
    def measure_loop_diameter(skeleton_pixels: Set[Tuple[int, int]]) -> float:
        """Returns max Chebyshev distance (max max(|r1-r2|, |c1-c2|)) among all skeletal nodes."""
        if not skeleton_pixels:
            return 0.0

        max_dist = 0
        pixels_list = list(skeleton_pixels)
        n = len(pixels_list)

        for i in range(n):
            r1, c1 = pixels_list[i]
            for j in range(i + 1, n):
                r2, c2 = pixels_list[j]
                dist = max(abs(r1 - r2), abs(c1 - c2))
                if dist > max_dist:
                    max_dist = dist

        return float(max_dist)
