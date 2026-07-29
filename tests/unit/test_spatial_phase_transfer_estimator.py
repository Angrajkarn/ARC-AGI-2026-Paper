"""
Unit tests for SpatialPhaseTransferEstimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.spatial_phase_transfer_estimator import SpatialPhaseTransferEstimator


class TestSpatialPhaseTransferEstimator:
    def test_compute_transfer_function_magnitude_diff_shape(self):
        grid1 = ArcGrid(pixels=np.ones((2, 2), dtype=np.uint8), background=0)
        grid2 = ArcGrid(pixels=np.ones((3, 3), dtype=np.uint8), background=0)

        tf = SpatialPhaseTransferEstimator.compute_transfer_function_magnitude(grid1, grid2)
        assert tf == 0.0

    def test_compute_transfer_function_magnitude_identical(self):
        grid1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]], dtype=np.uint8), background=0)
        grid2 = ArcGrid(pixels=np.array([[1, 2], [3, 4]], dtype=np.uint8), background=0)

        tf = SpatialPhaseTransferEstimator.compute_transfer_function_magnitude(grid1, grid2)
        # H(u,v) = Y/X = 1.0 everywhere -> mean = 1.0
        assert pytest.approx(tf, abs=1e-3) == 1.0
