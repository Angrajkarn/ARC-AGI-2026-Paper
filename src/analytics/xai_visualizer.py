"""
XAIProgramVisualizer — Computes pixel-level difference heatmaps showing spatial impact of transformations.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class XAIProgramVisualizer:
    """Computes difference heatmaps highlighting pixel regions transformed by programs."""

    @staticmethod
    def compute_heatmap(input_grid: ArcGrid, output_grid: ArcGrid) -> np.ndarray:
        """Returns binary matrix where 1 represents modified pixels, 0 represents unchanged."""
        if input_grid.height != output_grid.height or input_grid.width != output_grid.width:
            # Dimension mismatch: entire output grid is modified relative to input
            return np.ones((output_grid.height, output_grid.width), dtype=np.uint8)

        heatmap = (input_grid.pixels != output_grid.pixels).astype(np.uint8)
        return heatmap
