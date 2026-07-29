"""
SkeletonBettiBoundaryEstimator — Evaluates persistence lifespans of 1-cycle boundaries on skeletal graphs.
"""

from __future__ import annotations

from typing import Set, Tuple


class SkeletonBettiBoundaryEstimator:
    """Computes persistence lifespan for boundary loops on 2D grid graph skeletons."""

    @staticmethod
    def compute_betti_boundary_persistence(skeleton_pixels: Set[Tuple[int, int]]) -> float:
        """Returns persistence score based on skeletal loop perimeter to diameter ratio."""
        if not skeleton_pixels:
            return 0.0

        # Calculate bounding dimensions
        rows = [r for r, c in skeleton_pixels]
        cols = [c for r, c in skeleton_pixels]

        height = max(rows) - min(rows) + 1
        width = max(cols) - min(cols) + 1
        bounding_perimeter = 2.0 * (height + width)

        # Persistence score is boundary ratio
        score = float(len(skeleton_pixels)) / float(bounding_perimeter)
        return float(min(score, 10.0))
