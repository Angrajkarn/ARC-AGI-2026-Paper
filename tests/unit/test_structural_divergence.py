"""
Unit tests for StructuralDivergenceEvaluator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.structural_divergence import StructuralDivergenceEvaluator


class TestStructuralDivergenceEvaluator:
    def test_compute_divergence_same(self):
        grid1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]], dtype=np.uint8), background=0)
        grid2 = ArcGrid(pixels=np.array([[1, 2], [3, 4]], dtype=np.uint8), background=0)

        div = StructuralDivergenceEvaluator.compute_divergence(grid1, grid2)
        assert div == 0.0

    def test_compute_divergence_diff_size(self):
        grid1 = ArcGrid(pixels=np.zeros((2, 2), dtype=np.uint8), background=0)
        grid2 = ArcGrid(pixels=np.zeros((3, 3), dtype=np.uint8), background=0)

        div = StructuralDivergenceEvaluator.compute_divergence(grid1, grid2)
        assert div == 1.0

    def test_compute_divergence_half(self):
        grid1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]], dtype=np.uint8), background=0)
        grid2 = ArcGrid(pixels=np.array([[1, 5], [3, 5]], dtype=np.uint8), background=0)

        # 2 / 4 pixels differ -> 0.5 divergence
        div = StructuralDivergenceEvaluator.compute_divergence(grid1, grid2)
        assert div == 0.5
