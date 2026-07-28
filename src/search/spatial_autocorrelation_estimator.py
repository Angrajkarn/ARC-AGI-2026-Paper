"""
SpatialAutocorrelationEstimator — Computes 2D spatial autocorrelation profiles on grid patterns.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class SpatialAutocorrelationEstimator:
    """Estimates spatial autocorrelation and texture self-similarity."""

    @staticmethod
    def compute_autocorrelation(grid: ArcGrid, shift: tuple[int, int] = (1, 0)) -> float:
        """Returns normalized 2D autocorrelation score for a given row/col shift offset."""
        pixels = grid.pixels.astype(float)
        norm = np.linalg.norm(pixels)
        if norm == 0.0:
            return 1.0

        shifted = np.roll(pixels, shift=shift, axis=(0, 1))
        corr = np.sum(pixels * shifted) / (norm * norm)
        return float(np.clip(corr, -1.0, 1.0))
