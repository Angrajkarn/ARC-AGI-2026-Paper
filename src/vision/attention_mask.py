"""
VisualAttentionMask — generates visual coordinate attention masks prioritizing high-activity boundaries and object interfaces.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class VisualAttentionMask:
    """Computes coordinate attention weight masks focusing on high-density interface zones."""

    @staticmethod
    def compute_boundary_mask(grid: ArcGrid) -> np.ndarray:
        """Returns binary mask (height, width) containing 1s at non-background boundary interfaces."""
        h, w = grid.height, grid.width
        mask = np.zeros((h, w), dtype=np.uint8)

        for r in range(h):
            for c in range(w):
                if grid.get(r, c) != grid.background:
                    # Check if adjacent to background
                    is_boundary = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if grid.get(nr, nc) == grid.background:
                                is_boundary = True
                                break
                        else:
                            # Edge of grid counts as boundary
                            is_boundary = True
                            break

                    if is_boundary:
                        mask[r, c] = 1

        return mask
