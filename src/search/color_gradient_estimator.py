"""
ColorGradientEstimator — Detects if color values form a sequence gradient along axes.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class ColorGradientEstimator:
    """Estimates prior color gradient properties on active regions."""

    @staticmethod
    def is_linear_gradient(grid: ArcGrid) -> bool:
        """Returns True if the grid rows or columns show strictly increasing or decreasing color values (excluding background)."""
        pixels = grid.pixels
        h, w = pixels.shape

        if h < 3 and w < 3:
            return False

        # Check rows for gradient
        row_gradients = []
        for r in range(h):
            row_vals = [int(val) for val in pixels[r, :] if val != grid.background]
            if len(row_vals) >= 3:
                diffs = np.diff(row_vals)
                # Strictly increasing or decreasing
                if np.all(diffs > 0) or np.all(diffs < 0):
                    row_gradients.append(True)

        if len(row_gradients) >= h // 2 and len(row_gradients) > 0:
            return True

        # Check columns for gradient
        col_gradients = []
        for c in range(w):
            col_vals = [int(val) for val in pixels[:, c] if val != grid.background]
            if len(col_vals) >= 3:
                diffs = np.diff(col_vals)
                if np.all(diffs > 0) or np.all(diffs < 0):
                    col_gradients.append(True)

        if len(col_gradients) >= w // 2 and len(col_gradients) > 0:
            return True

        return False
