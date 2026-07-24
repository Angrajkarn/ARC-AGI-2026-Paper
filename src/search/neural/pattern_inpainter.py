"""
PatternInpainter — Autocompletes empty/background cells using reflection symmetry axes.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class PatternInpainter:
    """Reconstructs missing pixel values by reflecting across horizontal or vertical lines of symmetry."""

    @staticmethod
    def autocomplete_grid(grid: ArcGrid) -> ArcGrid:
        """Autocompletes empty grid values using horizontal reflection."""
        pixels = grid.pixels.copy()
        h, w = pixels.shape
        bg = grid.background

        # Attempt vertical axis reflection: mirror left half onto right half
        half_w = w // 2
        for r in range(h):
            for c in range(half_w):
                mirrored_c = w - 1 - c
                if pixels[r, mirrored_c] == bg and pixels[r, c] != bg:
                    pixels[r, mirrored_c] = pixels[r, c]
                elif pixels[r, c] == bg and pixels[r, mirrored_c] != bg:
                    pixels[r, c] = pixels[r, mirrored_c]

        return ArcGrid(pixels=pixels, background=bg)
