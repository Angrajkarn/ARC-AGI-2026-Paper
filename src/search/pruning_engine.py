"""
PruningEngine — Performs search space branch pruning using cost-to-go estimators.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class PruningEngine:
    """Computes lower bound cost estimations to prune suboptimal search paths."""

    @staticmethod
    def estimate_cost_to_go(current: ArcGrid, target: ArcGrid) -> int:
        """Estimates the minimum number of pixel edits or operations needed to match target."""
        if current.height != target.height or current.width != target.width:
            # Different sizes require resizing/cropping (count as at least 1 operation)
            return 2

        # Count number of mismatched pixels
        mismatch_count = np.sum(current.pixels != target.pixels)
        if mismatch_count == 0:
            return 0

        # An operation affects multiple pixels (e.g. rotation, translation, color replace)
        # We assume 1 DSL primitive can fix at most N mismatched pixels.
        # Thus, cost-to-go lower bound is at least 1 operation.
        return 1

    def should_prune(
        self, current: ArcGrid, target: ArcGrid, depth: int, max_budget: int
    ) -> bool:
        """Prunes search node if current depth + cost-to-go exceeds maximum allowed budget."""
        cost_to_go = self.estimate_cost_to_go(current, target)
        return (depth + cost_to_go) > max_budget
