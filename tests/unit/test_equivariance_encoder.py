"""
Unit tests for EquivarianceEncoder.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.core.objects.detector import ObjectDetector
from src.search.neural.equivariance_encoder import EquivarianceEncoder


class TestEquivarianceEncoder:
    def test_invariant_hash_rotations(self):
        # A simple horizontal 1x3 line shape grid
        grid1 = ArcGrid(pixels=np.array([[3, 3, 3], [0, 0, 0], [0, 0, 0]]), background=0)
        # A vertical 3x1 line shape grid (rotated version)
        grid2 = ArcGrid(pixels=np.array([[3, 0, 0], [3, 0, 0], [3, 0, 0]]), background=0)

        detector = ObjectDetector()
        obj1 = detector.detect(grid1)[0]
        obj2 = detector.detect(grid2)[0]

        hash1 = EquivarianceEncoder.compute_invariant_hash(obj1)
        hash2 = EquivarianceEncoder.compute_invariant_hash(obj2)

        # Both shapes represent congruent line shapes under D4, so hashes should be identical
        assert hash1 == hash2
