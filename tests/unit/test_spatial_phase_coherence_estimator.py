"""
Unit tests for SpatialPhaseCoherenceEstimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.spatial_phase_coherence_estimator import SpatialPhaseCoherenceEstimator


class TestSpatialPhaseCoherenceEstimator:
    def test_compute_phase_coherence_identical(self):
        grid1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]], dtype=np.uint8), background=0)
        grid2 = ArcGrid(pixels=np.array([[1, 2], [3, 4]], dtype=np.uint8), background=0)

        coh = SpatialPhaseCoherenceEstimator.compute_phase_coherence(grid1, grid2)
        assert pytest.approx(coh) == 1.0

    def test_compute_phase_coherence_different_shape(self):
        grid1 = ArcGrid(pixels=np.ones((2, 2), dtype=np.uint8), background=0)
        grid2 = ArcGrid(pixels=np.ones((3, 3), dtype=np.uint8), background=0)

        coh = SpatialPhaseCoherenceEstimator.compute_phase_coherence(grid1, grid2)
        assert coh == 0.0
