"""
RefinementDiscriminator — Heuristically scores grid prediction sanity and noise levels.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class RefinementDiscriminator:
    """Evaluates local boundaries to detect isolated pixel noise or coloring artifacts."""

    @staticmethod
    def score_grid_sanity(grid: ArcGrid) -> float:
        """Assigns a score from 0.0 (high noise/anomalies) to 1.0 (smooth, structured grid)."""
        pixels = grid.pixels
        h, w = pixels.shape
        bg = grid.background

        noise_count = 0
        total_non_bg = 0

        for r in range(h):
            for c in range(w):
                val = pixels[r, c]
                if val != bg:
                    total_non_bg += 1
                    # Inspect 4-way orthogonal neighbors
                    n_count = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if pixels[nr, nc] == val:
                                n_count += 1
                    # If pixel is isolated (0 matching neighbors), count as potential noise artifact
                    if n_count == 0:
                        noise_count += 1

        if total_non_bg == 0:
            return 1.0

        noise_ratio = noise_count / total_non_bg
        return float(np.clip(1.0 - noise_ratio, 0.0, 1.0))
