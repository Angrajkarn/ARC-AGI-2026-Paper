"""
Unit tests for SpatialSpectralDensityEstimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.spatial_spectral_density_estimator import SpatialSpectralDensityEstimator


class TestSpatialSpectralDensityEstimator:
    def test_compute_spectral_density_diff_shape(self):
        grid1 = ArcGrid(pixels=np.ones((2, 2), dtype=np.uint8), background=0)
        grid2 = ArcGrid(pixels=np.ones((3, 3), dtype=np.uint8), background=0)

        density = SpatialSpectralDensityEstimator.compute_spectral_density(grid1, grid2)
        assert density == 0.0

    def test_compute_spectral_density_same(self):
        grid1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]], dtype=np.uint8), background=0)
        grid2 = ArcGrid(pixels=np.array([[1, 2], [3, 4]], dtype=np.uint8), background=0)

        density = SpatialSpectralDensityEstimator.compute_spectral_density(grid1, grid2)
        assert density > 0.0
