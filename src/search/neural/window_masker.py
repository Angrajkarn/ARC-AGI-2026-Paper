"""
WindowFocusMasker — Identifies active window bounding boxes of non-background sub-grid patterns.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.core.grid.grid import ArcGrid


class WindowFocusMasker:
    """Finds tightest crop box enclosing all non-background elements."""

    @staticmethod
    def get_focus_window(grid: ArcGrid) -> Tuple[int, int, int, int]:
        """Returns bounding box (min_row, max_row, min_col, max_col) of non-background items."""
        pixels = grid.pixels
        bg = grid.background

        mask = pixels != bg
        r_indices, c_indices = np.where(mask)

        if r_indices.size == 0:
            return 0, 0, 0, 0

        return (
            int(np.min(r_indices)),
            int(np.max(r_indices)),
            int(np.min(c_indices)),
            int(np.max(c_indices)),
        )
