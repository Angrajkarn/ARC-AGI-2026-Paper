"""
Unit tests for CurriculumMetricAnalyzer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.meta_learning.curriculum_metric import CurriculumMetricAnalyzer


class TestCurriculumMetricAnalyzer:
    def test_calculate_entropy_zero(self):
        # Monochromatic grid (all 0s) -> Entropy should be 0.0
        pixels = np.zeros((4, 4), dtype=np.uint8)
        grid = ArcGrid(pixels=pixels, background=0)

        entropy = CurriculumMetricAnalyzer.calculate_entropy(grid)
        assert entropy == 0.0

    def test_calculate_entropy_uniform(self):
        # Grid with exactly half color 1 and half color 2 -> Entropy should be 1.0
        pixels = np.array([[1, 2], [1, 2]], dtype=np.uint8)
        grid = ArcGrid(pixels=pixels, background=0)

        entropy = CurriculumMetricAnalyzer.calculate_entropy(grid)
        assert pytest.approx(entropy) == 1.0
