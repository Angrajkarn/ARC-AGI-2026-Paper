"""
Unit tests for SpatialCorrelationEstimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.spatial_correlation_estimator import SpatialCorrelationEstimator


class TestSpatialCorrelationEstimator:
    def test_compute_correlation_perfect(self):
        grid1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]], dtype=np.uint8), background=0)
        grid2 = ArcGrid(pixels=np.array([[1, 2], [3, 4]], dtype=np.uint8), background=0)

        corr = SpatialCorrelationEstimator.compute_correlation(grid1, grid2)
        assert pytest.approx(corr) == 1.0

    def test_compute_correlation_orthogonal(self):
        grid1 = ArcGrid(pixels=np.array([[1, 0], [0, 0]], dtype=np.uint8), background=0)
        grid2 = ArcGrid(pixels=np.array([[0, 2], [3, 4]], dtype=np.uint8), background=0)

        corr = SpatialCorrelationEstimator.compute_correlation(grid1, grid2)
        assert corr == 0.0
