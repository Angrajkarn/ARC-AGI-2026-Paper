"""
Unit tests for TopologyInvariantMatcher.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.reflection.topology_matcher import TopologyInvariantMatcher


class TestTopologyMatcher:
    def test_solid_square_signature(self):
        # A solid square: Betti-0 = 1, Betti-1 = 0
        pixels = np.zeros((5, 5), dtype=np.uint8)
        pixels[1:4, 1:4] = 1
        grid = ArcGrid(pixels=pixels, background=0)

        matcher = TopologyInvariantMatcher()
        sig = matcher.compute_signature(grid)

        assert sig.betti_0 == 1
        assert sig.betti_1 == 0
        assert sig.euler_characteristic == 1

    def test_donut_signature(self):
        # Donut shape: hollow center (hole): Betti-0 = 1, Betti-1 = 1
        pixels = np.zeros((5, 5), dtype=np.uint8)
        pixels[1:4, 1:4] = 1
        pixels[2, 2] = 0  # Enclosed hole at center
        grid = ArcGrid(pixels=pixels, background=0)

        matcher = TopologyInvariantMatcher()
        sig = matcher.compute_signature(grid)

        assert sig.betti_0 == 1
        assert sig.betti_1 == 1
        assert sig.euler_characteristic == 0
