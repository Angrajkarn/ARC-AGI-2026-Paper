"""
Unit tests for GroupConvPrior.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.neural.group_conv import GroupConvPrior


class TestGroupConv:
    def test_apply_equivariant_conv(self):
        # A simple symmetric center pixel grid
        pixels = np.zeros((5, 5), dtype=np.uint8)
        pixels[2, 2] = 9
        grid = ArcGrid(pixels=pixels, background=0)

        prior = GroupConvPrior(kernel_size=3)
        feature_map = prior.apply_equivariant_conv(grid)

        # Output feature map should be 5x5
        assert feature_map.shape == (5, 5)
        # Center element should have highest value
        assert feature_map[2, 2] > feature_map[0, 0]
