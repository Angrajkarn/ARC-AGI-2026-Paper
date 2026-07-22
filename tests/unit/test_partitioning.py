"""
Unit tests for TessellationMatcher.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.vision.partitioning import TessellationMatcher


class TestTessellationMatcher:
    def test_find_split_lines(self):
        # A 5x5 grid divided by a center row of color 8
        pixels = np.zeros((5, 5), dtype=np.uint8)
        pixels[2, :] = 8
        grid = ArcGrid(pixels=pixels, background=0)

        h_lines, v_lines = TessellationMatcher.find_split_lines(grid)
        assert h_lines == [2]
        assert v_lines == []
