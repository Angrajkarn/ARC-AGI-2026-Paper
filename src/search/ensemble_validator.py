"""
MultiModelLLMValidator — performs consensus voting over candidates from multiple models or search configs.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional

from src.core.grid.grid import ArcGrid


class MultiModelLLMValidator:
    """Combines and votes on candidate predictions from multiple models or solver runs."""

    @staticmethod
    def get_consensus(candidates: List[ArcGrid]) -> Optional[ArcGrid]:
        """Runs majority voting on grid candidates, returns the most frequent grid pattern."""
        if not candidates:
            return None

        # Hash each candidate's pixel state to count occurrences
        hex_hashes = [grid.pixels.tobytes().hex() for grid in candidates]
        counter = Counter(hex_hashes)

        # Get most common grid hex representation
        most_common_hex, count = counter.most_common(1)[0]

        # Find corresponding ArcGrid instance
        for grid in candidates:
            if grid.pixels.tobytes().hex() == most_common_hex:
                return grid

        return None
