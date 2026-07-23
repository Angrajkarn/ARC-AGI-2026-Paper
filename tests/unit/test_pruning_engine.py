"""
Unit tests for PruningEngine.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.pruning_engine import PruningEngine


class TestPruningEngine:
    def test_estimate_cost_to_go(self):
        g1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        g2 = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        g3 = ArcGrid(pixels=np.array([[1, 9], [3, 4]]), background=0)

        engine = PruningEngine()
        assert engine.estimate_cost_to_go(g1, g2) == 0
        assert engine.estimate_cost_to_go(g1, g3) == 1

    def test_should_prune(self):
        g1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        g3 = ArcGrid(pixels=np.array([[1, 9], [3, 4]]), background=0)

        engine = PruningEngine()
        # Depth 2 + Cost-to-go 1 = 3 (exceeds budget 2) -> Should prune
        assert engine.should_prune(g1, g3, depth=2, max_budget=2) is True
        # Depth 1 + Cost-to-go 1 = 2 (within budget 2) -> Should not prune
        assert engine.should_prune(g1, g3, depth=1, max_budget=2) is False
