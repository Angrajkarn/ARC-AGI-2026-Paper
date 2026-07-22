"""
ProgramSlicing — decomposes grid arrays into independent color channel layers and overlays them.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.core.grid.grid import ArcGrid


class ProgramSlicing:
    """Slices grids into single-color channel masks for multi-channel sub-program synthesis."""

    @staticmethod
    def slice_channels(grid: ArcGrid) -> Dict[int, ArcGrid]:
        """Returns map from color to sliced ArcGrid channel containing only that color."""
        sliced: Dict[int, ArcGrid] = {}
        for color in grid.colors:
            if color == grid.background:
                continue

            # Create binary-like mask grid containing only target color pixels
            channel_pixels = np.full_like(grid.pixels, grid.background)
            mask = grid.pixels == color
            channel_pixels[mask] = color

            sliced[color] = ArcGrid(pixels=channel_pixels, background=grid.background)

        return sliced

    @staticmethod
    def merge_channels(channels: List[ArcGrid], background: int = 0) -> ArcGrid:
        """Overlays a list of grid channels back into a single unified grid."""
        if not channels:
            # Return standard empty grid shape
            return ArcGrid(pixels=np.zeros((1, 1), dtype=np.uint8), background=background)

        ref = channels[0]
        merged_pixels = np.full_like(ref.pixels, background)

        # Overlay non-background pixels onto merged canvas
        for ch in channels:
            mask = ch.pixels != ch.background
            merged_pixels[mask] = ch.pixels[mask]

        return ArcGrid(pixels=merged_pixels, background=background)
