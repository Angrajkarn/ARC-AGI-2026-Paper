"""
Unit tests for SpatialFrequencyEstimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.spatial_frequency_estimator import SpatialFrequencyEstimator


class TestSpatialFrequencyEstimator:
    def test_estimate_spatial_frequency_zero(self):
        # Uniform grid -> 0 transitions -> 0.0 frequency
        pixels = np.zeros((3, 3), dtype=np.uint8)
        grid = ArcGrid(pixels=pixels, background=0)

        freq = SpatialFrequencyEstimator.estimate_spatial_frequency(grid)
        assert freq == 0.0

    def test_estimate_spatial_frequency_checkerboard(self):
        # Checkerboard -> max transitions -> 1.0 frequency
        pixels = np.array([
            [1, 2],
            [2, 1]
        ], dtype=np.uint8)
        grid = ArcGrid(pixels=pixels, background=0)

        freq = SpatialFrequencyEstimator.estimate_spatial_frequency(grid)
        assert freq == 1.0
