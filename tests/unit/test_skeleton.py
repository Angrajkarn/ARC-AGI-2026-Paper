"""
Unit tests for TopologySkeletonizer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.vision.skeleton import TopologySkeletonizer


class TestTopologySkeletonizer:
    def test_extract_skeleton(self):
        # A solid 3x3 square of pixels
        pixels = np.zeros((5, 5), dtype=np.uint8)
        pixels[1:4, 1:4] = 1
        grid = ArcGrid(pixels=pixels, background=0)

        skeleton = TopologySkeletonizer.extract_skeleton(grid)
        # Skeletons should be thinner than original solid block (less than 9 pixels)
        active_pixel_count = (skeleton.pixels != skeleton.background).sum()
        assert active_pixel_count < 9
