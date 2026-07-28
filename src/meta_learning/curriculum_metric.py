"""
CurriculumMetricAnalyzer — Analyzes grid pattern entropy and unique color distributions.
"""

from __future__ import annotations

import math
from typing import Dict

from src.core.grid.grid import ArcGrid


class CurriculumMetricAnalyzer:
    """Calculates structural entropy stats for curriculum task ordering."""

    @staticmethod
    def calculate_entropy(grid: ArcGrid) -> float:
        """Returns Shannon entropy score based on color frequency counts."""
        total_pixels = grid.pixels.size
        if total_pixels == 0:
            return 0.0

        # Count occurrences of each color value
        from collections import Counter
        counts = Counter(grid.pixels.flat)

        entropy = 0.0
        for count in counts.values():
            p = float(count) / float(total_pixels)
            entropy -= p * math.log2(p)

        return entropy
