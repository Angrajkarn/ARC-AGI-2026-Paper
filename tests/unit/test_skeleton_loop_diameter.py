"""
Unit tests for SkeletonLoopDiameterEstimator.
"""

from __future__ import annotations

import pytest

from src.core.graphs.skeleton_loop_diameter import SkeletonLoopDiameterEstimator


class TestSkeletonLoopDiameterEstimator:
    def test_measure_loop_diameter_empty(self):
        diam = SkeletonLoopDiameterEstimator.measure_loop_diameter(set())
        assert diam == 0.0

    def test_measure_loop_diameter_box(self):
        # 3x3 hollow box: corners at (0, 0) and (2, 2) -> max distance is 2
        pixels = {
            (0, 0), (0, 1), (0, 2),
            (1, 0),         (1, 2),
            (2, 0), (2, 1), (2, 2)
        }
        diam = SkeletonLoopDiameterEstimator.measure_loop_diameter(pixels)
        assert diam == 2.0
