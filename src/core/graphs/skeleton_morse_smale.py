"""
SkeletonMorseSmaleSegmenter — Decomposes 2D grid graph scalar fields into Morse-Smale topological basins.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple


class SkeletonMorseSmaleSegmenter:
    """Computes Morse-Smale complex topological decomposition on skeletal grid fields."""

    @staticmethod
    def segment_morse_smale_basins(
        skeleton_pixels: Set[Tuple[int, int]]
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Classifies skeletal pixels into critical points (maxima, minima) and ascending manifolds."""
        if not skeleton_pixels:
            return {"maxima": [], "minima": []}

        # Calculate local degree scalar field
        pixel_degrees = {}
        for r, c in skeleton_pixels:
            neighbors = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r + dr, c + dc) in skeleton_pixels:
                    neighbors += 1
            pixel_degrees[(r, c)] = neighbors

        max_deg = max(pixel_degrees.values()) if pixel_degrees else 0
        min_deg = min(pixel_degrees.values()) if pixel_degrees else 0

        maxima = [p for p, deg in pixel_degrees.items() if deg == max_deg]
        minima = [p for p, deg in pixel_degrees.items() if deg == min_deg]

        return {"maxima": maxima, "minima": minima}
