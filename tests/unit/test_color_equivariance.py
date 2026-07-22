"""
Unit tests for ColorEquivarianceMatcher.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.vision.features.color_equivariance import ColorEquivarianceMatcher


class TestColorEquivariance:
    def test_find_mapping_success(self):
        # Maps color 1 -> 3
        g1 = ArcGrid(pixels=np.array([[1, 0], [0, 1]]), background=0)
        g2 = ArcGrid(pixels=np.array([[3, 0], [0, 3]]), background=0)

        mapping = ColorEquivarianceMatcher.find_mapping([{"input": g1, "output": g2}])
        assert mapping is not None
        assert mapping[1] == 3

    def test_find_mapping_failure(self):
        # Conflicting mappings for color 1 -> 3 and 1 -> 4
        g1 = ArcGrid(pixels=np.array([[1, 0], [0, 1]]), background=0)
        g2 = ArcGrid(pixels=np.array([[3, 0], [0, 4]]), background=0)

        mapping = ColorEquivarianceMatcher.find_mapping([{"input": g1, "output": g2}])
        assert mapping is None
