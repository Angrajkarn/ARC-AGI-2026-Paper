"""
Unit tests for RelaxedPruner.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.pruning_relaxed import RelaxedPruner


class TestRelaxedPruner:
    def test_should_prune_true(self):
        # Candidate has color 2, target has color 5 (disjoint, prune it!)
        candidate = ArcGrid(pixels=np.array([[2, 0], [0, 2]], dtype=np.uint8), background=0)
        target = ArcGrid(pixels=np.array([[5, 0], [0, 5]], dtype=np.uint8), background=0)

        assert RelaxedPruner.should_prune(candidate, target)

    def test_should_prune_false(self):
        # Candidate has color 2, target has color 2 (keep it!)
        candidate = ArcGrid(pixels=np.array([[2, 0], [0, 2]], dtype=np.uint8), background=0)
        target = ArcGrid(pixels=np.array([[2, 0], [0, 2]], dtype=np.uint8), background=0)

        assert not RelaxedPruner.should_prune(candidate, target)
