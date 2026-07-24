"""
Unit tests for ColorTopologyPrior.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.neural.color_topology import ColorTopologyPrior


class TestColorTopologyPrior:
    def test_get_color_topology(self):
        pixels = np.zeros((4, 4), dtype=np.uint8)
        pixels[1, 1] = 4
        pixels[2, 2] = 4

        grid = ArcGrid(pixels=pixels, background=0)
        topology = ColorTopologyPrior.get_color_topology(grid)

        color_4_box = topology.get(4)
        assert color_4_box is not None
        assert color_4_box == (2, 2)
