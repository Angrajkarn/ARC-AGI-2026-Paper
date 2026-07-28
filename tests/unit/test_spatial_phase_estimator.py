"""
Unit tests for SpatialPhaseEstimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.spatial_phase_estimator import SpatialPhaseEstimator


class TestSpatialPhaseEstimator:
    def test_compute_phase_shift_zero(self):
        # Grids are identical -> shift is (0, 0)
        grid1 = ArcGrid(pixels=np.array([[1, 0], [0, 2]], dtype=np.uint8), background=0)
        grid2 = ArcGrid(pixels=np.array([[1, 0], [0, 2]], dtype=np.uint8), background=0)

        shift = SpatialPhaseEstimator.compute_phase_shift(grid1, grid2)
        assert shift == (0, 0)

    def test_compute_phase_shift_translated(self):
        # Grid2 is Grid1 rolled down by 1 row
        grid1 = ArcGrid(pixels=np.array([[1, 0], [0, 0]], dtype=np.uint8), background=0)
        grid2 = ArcGrid(pixels=np.array([[0, 0], [1, 0]], dtype=np.uint8), background=0)

        shift = SpatialPhaseEstimator.compute_phase_shift(grid1, grid2)
        assert shift == (1, 0)
