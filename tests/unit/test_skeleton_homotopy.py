"""
Unit tests for SkeletonHomotopyExtractor.
"""

from __future__ import annotations

import pytest

from src.core.graphs.skeleton_homotopy import SkeletonHomotopyExtractor


class TestSkeletonHomotopyExtractor:
    def test_extract_homotopy_generators_empty(self):
        rank = SkeletonHomotopyExtractor.extract_homotopy_generators(set())
        assert rank == 0

    def test_extract_homotopy_generators_loop(self):
        # 3x3 hollow box loop -> rank of pi_1 = 1 generator
        pixels = {
            (0, 0), (0, 1), (0, 2),
            (1, 0),         (1, 2),
            (2, 0), (2, 1), (2, 2)
        }
        rank = SkeletonHomotopyExtractor.extract_homotopy_generators(pixels)
        assert rank == 1
