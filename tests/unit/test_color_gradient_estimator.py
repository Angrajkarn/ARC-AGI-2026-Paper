"""
Unit tests for ColorGradientEstimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.color_gradient_estimator import ColorGradientEstimator


class TestColorGradientEstimator:
    def test_is_linear_gradient_true(self):
        # Rows with increasing color numbers: [1, 2, 3]
        pixels = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]
        ], dtype=np.uint8)
        grid = ArcGrid(pixels=pixels, background=0)

        assert ColorGradientEstimator.is_linear_gradient(grid)

    def test_is_linear_gradient_false(self):
        # Non-monotone colors: [1, 3, 2]
        pixels = np.array([
            [1, 3, 2],
            [1, 3, 2],
            [1, 3, 2]
        ], dtype=np.uint8)
        grid = ArcGrid(pixels=pixels, background=0)

        assert not ColorGradientEstimator.is_linear_gradient(grid)
