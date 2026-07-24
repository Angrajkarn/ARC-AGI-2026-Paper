"""
RelaxedPruner — Prunes candidate program search spaces with target color distribution divergence checking.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class RelaxedPruner:
    """Evaluates candidates using relaxed continuous grid comparisons to prune search spaces."""

    @staticmethod
    def should_prune(candidate: ArcGrid, target: ArcGrid) -> bool:
        """Prunes candidate grid if unique target colors do not overlap with candidate's colors."""
        target_colors = set(target.pixels.flat)
        candidate_colors = set(candidate.pixels.flat)

        # Exclude background color
        target_colors.discard(target.background)
        candidate_colors.discard(candidate.background)

        # If target has colors not present in candidate, prune it (cannot be the solution)
        if target_colors and not target_colors.intersection(candidate_colors):
            return True

        return False
