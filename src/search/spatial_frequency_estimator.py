"""
SpatialFrequencyEstimator — Analyzes 2D spatial frequency signatures on grid patterns.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class SpatialFrequencyEstimator:
    """Estimates spatial frequency and texture repetition density."""

    @staticmethod
    def estimate_spatial_frequency(grid: ArcGrid) -> float:
        """Returns average spatial transition density (number of adjacent color changes per unit area)."""
        pixels = grid.pixels
        h, w = pixels.shape
        if h <= 1 and w <= 1:
            return 0.0

        # Horizontal transitions
        h_diffs = np.sum(pixels[:, :-1] != pixels[:, 1:])
        # Vertical transitions
        v_diffs = np.sum(pixels[:-1, :] != pixels[1:, :])

        total_edges = (w - 1) * h + (h - 1) * w
        if total_edges == 0:
            return 0.0

        return float(h_diffs + v_diffs) / float(total_edges)
