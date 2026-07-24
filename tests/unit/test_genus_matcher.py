"""
Unit tests for GenusMatcher.
"""

from __future__ import annotations

import pytest

from src.core.graphs.genus_matcher import GenusMatcher


class TestGenusMatcher:
    def test_compute_genus_no_holes(self):
        # A solid 2x2 square object
        pixels = {(0, 0), (0, 1), (1, 0), (1, 1)}
        genus = GenusMatcher.compute_genus(pixels)
        assert genus == 0

    def test_compute_genus_with_hole(self):
        # A 3x3 hollow box (donut) has 1 hole in the center (1, 1)
        pixels = {
            (0, 0), (0, 1), (0, 2),
            (1, 0),         (1, 2),
            (2, 0), (2, 1), (2, 2)
        }
        genus = GenusMatcher.compute_genus(pixels)
        assert genus == 1
