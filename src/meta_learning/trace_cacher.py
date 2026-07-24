"""
TraceCacher — Caches grid state hashes during search to prevent redundant execution branches.
"""

from __future__ import annotations

from typing import Set

import numpy as np

from src.core.grid.grid import ArcGrid


class TraceCacher:
    """Manages intermediate execution trace history to prune cycles or redundant state generation."""

    def __init__(self) -> None:
        self.seen_hashes: Set[int] = set()

    def get_grid_hash(self, grid: ArcGrid) -> int:
        """Computes a deterministic hash of the grid's pixels."""
        # Using built-in hash on bytes
        return hash(grid.pixels.tobytes())

    def check_and_add(self, grid: ArcGrid) -> bool:
        """Returns True if the grid state has already been seen; otherwise stores it and returns False."""
        h = self.get_grid_hash(grid)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False
