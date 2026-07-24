"""
Unit tests for PatternInpainter.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.neural.pattern_inpainter import PatternInpainter


class TestPatternInpainter:
    def test_autocomplete_grid(self):
        # A grid with symmetry on the left but empty on the right
        pixels = np.zeros((4, 4), dtype=np.uint8)
        pixels[0, 0] = 7
        pixels[2, 1] = 5
        grid = ArcGrid(pixels=pixels, background=0)

        completed = PatternInpainter.autocomplete_grid(grid)

        # Right side should be completed with mirrored left side values
        assert completed.pixels[0, 3] == 7
        assert completed.pixels[2, 2] == 5
