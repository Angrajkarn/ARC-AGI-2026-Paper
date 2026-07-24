"""
ColorTopologyPrior — Compares topological component features of color grids.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid


class ColorTopologyPrior:
    """Computes layout bounding stats of color components for topological checks."""

    @staticmethod
    def get_color_topology(grid: ArcGrid) -> Dict[int, Tuple[int, int]]:
        """Returns height and width of each non-background color bounding box."""
        pixels = grid.pixels
        bg = grid.background
        unique_colors = np.unique(pixels)

        topology: Dict[int, Tuple[int, int]] = {}
        for color in unique_colors:
            if color == bg:
                continue

            r_indices, c_indices = np.where(pixels == color)
            if r_indices.size > 0:
                h_box = int(np.max(r_indices) - np.min(r_indices) + 1)
                w_box = int(np.max(c_indices) - np.min(c_indices) + 1)
                topology[int(color)] = (h_box, w_box)

        return topology
