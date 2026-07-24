"""
StructuralDivergenceEvaluator — Evaluates divergence scores between grid patterns.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class StructuralDivergenceEvaluator:
    """Computes structural grid similarity distance metrics."""

    @staticmethod
    def compute_divergence(grid1: ArcGrid, grid2: ArcGrid) -> float:
        """Returns structural color difference ratio between two grids."""
        if grid1.pixels.shape != grid2.pixels.shape:
            return 1.0  # Maximum structural difference

        diff_count = np.sum(grid1.pixels != grid2.pixels)
        total_pixels = grid1.pixels.size
        return float(diff_count) / float(total_pixels)
