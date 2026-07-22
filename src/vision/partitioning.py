"""
TessellationMatcher — detects boundary dividing lines and segments the grid into sub-panels.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid


class TessellationMatcher:
    """Finds uniform color horizontal/vertical splitting gridlines dividing grid panels."""

    @staticmethod
    def find_split_lines(grid: ArcGrid) -> Tuple[List[int], List[int]]:
        """Identifies horizontal row indices and vertical column indices acting as divider lines."""
        h, w = grid.height, grid.width
        h_lines: List[int] = []
        v_lines: List[int] = []

        # Find row indices filled with a single color
        for r in range(h):
            row_vals = grid.pixels[r, :]
            if len(np.unique(row_vals)) == 1 and row_vals[0] != grid.background:
                h_lines.append(r)

        # Find col indices filled with a single color
        for c in range(w):
            col_vals = grid.pixels[:, c]
            if len(np.unique(col_vals)) == 1 and col_vals[0] != grid.background:
                v_lines.append(c)

        return h_lines, v_lines
