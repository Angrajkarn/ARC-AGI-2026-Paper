"""
Unit tests for SkeletonLoopPerimeterDetector.
"""

from __future__ import annotations

import pytest

from src.core.graphs.skeleton_loop_perimeter import SkeletonLoopPerimeterDetector


class TestSkeletonLoopPerimeterDetector:
    def test_measure_loop_perimeter_square(self):
        # A 3x3 hollow square box (perimeter = 8)
        pixels = {
            (0, 0), (0, 1), (0, 2),
            (1, 0),         (1, 2),
            (2, 0), (2, 1), (2, 2)
        }

        perim = SkeletonLoopPerimeterDetector.measure_loop_perimeter(pixels)
        # All 8 boundary pixels have <4 orthogonal neighbors -> perim = 8
        assert perim == 8
