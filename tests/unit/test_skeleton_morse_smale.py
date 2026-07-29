"""
Unit tests for SkeletonMorseSmaleSegmenter.
"""

from __future__ import annotations

import pytest

from src.core.graphs.skeleton_morse_smale import SkeletonMorseSmaleSegmenter


class TestSkeletonMorseSmaleSegmenter:
    def test_segment_morse_smale_basins_empty(self):
        res = SkeletonMorseSmaleSegmenter.segment_morse_smale_basins(set())
        assert res == {"maxima": [], "minima": []}

    def test_segment_morse_smale_basins_line(self):
        # Line: (0,0)-(0,1)-(0,2) -> end points (0,0) and (0,2) have degree 1 (minima), middle has degree 2 (maxima)
        pixels = {(0, 0), (0, 1), (0, 2)}
        res = SkeletonMorseSmaleSegmenter.segment_morse_smale_basins(pixels)

        assert (0, 1) in res["maxima"]
        assert (0, 0) in res["minima"]
