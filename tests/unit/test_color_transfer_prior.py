"""
Unit tests for ColorTransferPrior.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.neural.color_transfer_prior import ColorTransferPrior


class TestColorTransferPrior:
    def test_discover_color_mapping(self):
        # src has color 3 (three times) and color 4 (two times)
        src_pixels = np.array([
            [3, 3, 3],
            [4, 4, 0]
        ], dtype=np.uint8)
        # dst has color 7 (three times) and color 8 (two times)
        dst_pixels = np.array([
            [7, 7, 7],
            [8, 8, 0]
        ], dtype=np.uint8)

        src = ArcGrid(pixels=src_pixels, background=0)
        dst = ArcGrid(pixels=dst_pixels, background=0)

        mapping = ColorTransferPrior.discover_color_mapping(src, dst)

        # Most frequent color in src (3) should map to most frequent in dst (7)
        assert mapping[3] == 7
        # Second most frequent (4) should map to (8)
        assert mapping[4] == 8
