"""
Unit tests for SubgridAttention.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.neural.subgrid_attention import SubgridAttention


class TestSubgridAttention:
    def test_compute_self_attention(self):
        # A 4x4 grid. We have four 3x3 patches.
        pixels = np.zeros((4, 4), dtype=np.uint8)
        pixels[0:3, 0:3] = 1  # First patch has all 1s
        grid = ArcGrid(pixels=pixels, background=0)

        attention = SubgridAttention.compute_self_attention(grid)

        # 4 patches -> 4x4 similarity attention matrix
        assert attention.shape == (4, 4)
        # Cosine self-similarity of first patch is 1.0
        assert pytest.approx(attention[0, 0]) == 1.0
