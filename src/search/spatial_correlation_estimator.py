"""
SpatialCorrelationEstimator — Computes 2D normalized cross-correlation matrices between grids.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class SpatialCorrelationEstimator:
    """Computes 2D normalized cross-correlation scores between grid outputs and targets."""

    @staticmethod
    def compute_correlation(grid1: ArcGrid, grid2: ArcGrid) -> float:
        """Returns normalized 2D cross-correlation score in [0.0, 1.0]."""
        if grid1.pixels.shape != grid2.pixels.shape:
            return 0.0

        p1 = grid1.pixels.astype(float)
        p2 = grid2.pixels.astype(float)

        norm1 = np.linalg.norm(p1)
        norm2 = np.linalg.norm(p2)

        if norm1 == 0.0 or norm2 == 0.0:
            return 1.0 if norm1 == norm2 else 0.0

        correlation = float(np.sum(p1 * p2) / (norm1 * norm2))
        return float(np.clip(correlation, 0.0, 1.0))
