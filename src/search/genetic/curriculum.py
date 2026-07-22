"""
TaskCurriculumEngine — Ranks and orders task demonstration pairs by topological and color complexity.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from src.core.grid.grid import ArcGrid


class TaskCurriculumEngine:
    """Sorts training grid demonstration pairs incrementally from easiest to hardest."""

    @staticmethod
    def _compute_complexity(grid: ArcGrid) -> float:
        # Combined complexity metric: entropy + grid size + unique color count
        counts = np.bincount(grid.pixels.ravel(), minlength=10)
        probs = counts / counts.sum()
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        area_factor = (grid.height * grid.width) / 900.0  # Normalize relative to 30x30
        color_factor = len(grid.colors) / 10.0

        return entropy + area_factor + color_factor

    def sort_pairs(self, train_pairs: List[Dict[str, ArcGrid]]) -> List[Dict[str, ArcGrid]]:
        """Returns sorted list of pairs, sorting from lowest complexity to highest."""
        if not train_pairs:
            return []

        # Sort by input grid complexity score
        sorted_pairs = sorted(
            train_pairs,
            key=lambda pair: self._compute_complexity(pair["input"])
        )
        return sorted_pairs
