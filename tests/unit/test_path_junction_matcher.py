"""
Unit tests for PathJunctionMatcher.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.vision.path_junction_matcher import PathJunctionMatcher


class TestPathJunctionMatcher:
    def test_classify_coordinates_junction(self):
        # Create a T-shaped skeleton (center coordinate (2,2) has 3 neighbors)
        pixels = np.zeros((5, 5), dtype=np.uint8)
        pixels[2, 1:4] = 1  # Horizontal line segment
        pixels[3, 2] = 1    # Vertical segment attaching underneath
        grid = ArcGrid(pixels=pixels, background=0)

        classes = PathJunctionMatcher.classify_coordinates(grid)
        # Center coordinate (2,2) should be classified as junction (3 neighbors)
        assert (2, 2) in classes["junctions"]
        # Endpoint ends: (2,1), (2,3), and (3,2) should be endpoints
        assert (2, 1) in classes["endpoints"]
        assert (2, 3) in classes["endpoints"]
        assert (3, 2) in classes["endpoints"]
