"""
Unit tests for SpatialBispectralDensityEstimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.spatial_bispectral_density_estimator import SpatialBispectralDensityEstimator


class TestSpatialBispectralDensityEstimator:
    def test_compute_bispectral_density_diff_shape(self):
        grid1 = ArcGrid(pixels=np.ones((2, 2), dtype=np.uint8), background=0)
        grid2 = ArcGrid(pixels=np.ones((3, 3), dtype=np.uint8), background=0)

        bisp = SpatialBispectralDensityEstimator.compute_bispectral_density(grid1, grid2)
        assert bisp == 0.0

    def test_compute_bispectral_density_same(self):
        grid1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]], dtype=np.uint8), background=0)
        grid2 = ArcGrid(pixels=np.array([[1, 2], [3, 4]], dtype=np.uint8), background=0)

        bisp = SpatialBispectralDensityEstimator.compute_bispectral_density(grid1, grid2)
        assert bisp > 0.0
