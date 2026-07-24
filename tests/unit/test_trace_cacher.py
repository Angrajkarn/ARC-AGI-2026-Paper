"""
Unit tests for TraceCacher.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.meta_learning.trace_cacher import TraceCacher


class TestTraceCacher:
    def test_cache_mechanism(self):
        cacher = TraceCacher()
        grid1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]], dtype=np.uint8), background=0)
        grid2 = ArcGrid(pixels=np.array([[1, 2], [3, 4]], dtype=np.uint8), background=0)
        grid3 = ArcGrid(pixels=np.array([[5, 6], [7, 8]], dtype=np.uint8), background=0)

        # First time seen -> returns False
        assert not cacher.check_and_add(grid1)
        # Second time seen (equivalent grid) -> returns True
        assert cacher.check_and_add(grid2)
        # Different grid -> returns False
        assert not cacher.check_and_add(grid3)
