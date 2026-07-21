"""
Unit tests for SemanticProgramCache.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.dsl.cache.program_cache import SemanticProgramCache


class TestSemanticProgramCache:
    def test_cache_deduplication(self):
        in_grid = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        out_grid = ArcGrid(pixels=np.array([[2, 1], [4, 3]]), background=0)

        cache = SemanticProgramCache()
        assert not cache.is_seen(in_grid, out_grid)

        cache.add(in_grid, out_grid)
        assert cache.is_seen(in_grid, out_grid)

        cache.clear()
        assert not cache.is_seen(in_grid, out_grid)
