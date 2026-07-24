"""
PaddingPrior — Discovers exact grid border margins/padding configurations.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid


class PaddingPrior:
    """Estimates the padding thickness of background colors around a task's active area."""

    @staticmethod
    def get_padding_offsets(grid: ArcGrid) -> Tuple[int, int, int, int]:
        """Returns (top, bottom, left, right) padding offsets of the background color."""
        pixels = grid.pixels
        bg = grid.background
        h, w = pixels.shape

        non_bg_mask = pixels != bg
        r_indices, c_indices = np.where(non_bg_mask)

        if r_indices.size == 0:
            return 0, 0, 0, 0

        min_r, max_r = int(np.min(r_indices)), int(np.max(r_indices))
        min_c, max_c = int(np.min(c_indices)), int(np.max(c_indices))

        top = min_r
        bottom = h - 1 - max_r
        left = min_c
        right = w - 1 - max_c

        return top, bottom, left, right
