"""
Unit tests for SpatialAutocorrelationEstimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.spatial_autocorrelation_estimator import SpatialAutocorrelationEstimator


class TestSpatialAutocorrelationEstimator:
    def test_compute_autocorrelation_uniform(self):
        # Uniform non-zero grid -> roll shift produces identical grid -> correlation = 1.0
        grid = ArcGrid(pixels=np.ones((3, 3), dtype=np.uint8), background=0)
        autocorr = SpatialAutocorrelationEstimator.compute_autocorrelation(grid, shift=(1, 0))
        assert pytest.approx(autocorr) == 1.0

    def test_compute_autocorrelation_zero(self):
        grid = ArcGrid(pixels=np.zeros((3, 3), dtype=np.uint8), background=0)
        autocorr = SpatialAutocorrelationEstimator.compute_autocorrelation(grid, shift=(1, 0))
        assert autocorr == 1.0
