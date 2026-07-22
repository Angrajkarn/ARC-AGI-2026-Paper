"""
Unit tests for VisualAttentionMask.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.vision.attention_mask import VisualAttentionMask


class TestAttentionMask:
    def test_compute_boundary_mask(self):
        # A single center pixel of color 1 inside 3x3 background
        pixels = np.zeros((3, 3), dtype=np.uint8)
        pixels[1, 1] = 1
        grid = ArcGrid(pixels=pixels, background=0)

        mask = VisualAttentionMask.compute_boundary_mask(grid)
        # Center pixel (1,1) touches background so it must have attention 1
        assert mask[1, 1] == 1
        # Corner pixels are background so they must have attention 0
        assert mask[0, 0] == 0
        assert mask[2, 2] == 0
