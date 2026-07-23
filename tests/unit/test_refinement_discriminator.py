"""
Unit tests for RefinementDiscriminator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.neural.refinement_discriminator import RefinementDiscriminator


class TestRefinementDiscriminator:
    def test_score_grid_sanity_clean(self):
        # A 3x3 square - solid shape (perfectly clean, no isolated pixel noise)
        pixels = np.zeros((5, 5), dtype=np.uint8)
        pixels[1:4, 1:4] = 2
        grid = ArcGrid(pixels=pixels, background=0)

        score = RefinementDiscriminator.score_grid_sanity(grid)
        # Should be 1.0 because every non-background pixel has matching neighbors
        assert score == 1.0

    def test_score_grid_sanity_noisy(self):
        # Grid with isolated pixels (noise)
        pixels = np.zeros((5, 5), dtype=np.uint8)
        pixels[1, 1] = 2
        pixels[3, 3] = 4
        grid = ArcGrid(pixels=pixels, background=0)

        score = RefinementDiscriminator.score_grid_sanity(grid)
        # Both pixels are isolated, noise ratio is 2/2 = 1.0, score should be 0.0
        assert score == 0.0
