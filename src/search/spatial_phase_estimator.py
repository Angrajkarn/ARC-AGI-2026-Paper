"""
SpatialPhaseEstimator — Computes horizontal and vertical phase shift offsets between periodic subgrids.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.core.grid.grid import ArcGrid


class SpatialPhaseEstimator:
    """Estimates phase translation offsets between subgrid patterns."""

    @staticmethod
    def compute_phase_shift(grid1: ArcGrid, grid2: ArcGrid) -> Tuple[int, int]:
        """Returns best (row_shift, col_shift) minimizing pixel difference between equal-sized grids."""
        if grid1.pixels.shape != grid2.pixels.shape:
            return 0, 0

        h, w = grid1.pixels.shape
        min_diff = float("inf")
        best_shift = (0, 0)

        for dr in range(h):
            for dc in range(w):
                shifted = np.roll(grid1.pixels, shift=(dr, dc), axis=(0, 1))
                diff = np.sum(shifted != grid2.pixels)
                if diff < min_diff:
                    min_diff = diff
                    best_shift = (dr, dc)

        return best_shift
