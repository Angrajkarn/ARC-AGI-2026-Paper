"""
Unit tests for PaddingPrior.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.neural.padding_prior import PaddingPrior


class TestPaddingPrior:
    def test_get_padding_offsets(self):
        # 6x6 grid with 2x2 active square in the center (top=2, bottom=2, left=2, right=2)
        pixels = np.zeros((6, 6), dtype=np.uint8)
        pixels[2:4, 2:4] = 3

        grid = ArcGrid(pixels=pixels, background=0)
        offsets = PaddingPrior.get_padding_offsets(grid)

        # Should match (2, 2, 2, 2)
        assert offsets == (2, 2, 2, 2)
