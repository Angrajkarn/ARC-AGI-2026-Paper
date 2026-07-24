"""
Unit tests for WindowFocusMasker.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.neural.window_masker import WindowFocusMasker


class TestWindowFocusMasker:
    def test_get_focus_window(self):
        # A 5x5 grid with active values inside (1, 1) to (3, 3) bounds
        pixels = np.zeros((5, 5), dtype=np.uint8)
        pixels[1, 1] = 8
        pixels[3, 3] = 4

        grid = ArcGrid(pixels=pixels, background=0)
        bounds = WindowFocusMasker.get_focus_window(grid)

        # Expected bounding coordinates: (1, 3, 1, 3)
        assert bounds == (1, 3, 1, 3)
