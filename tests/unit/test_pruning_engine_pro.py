"""
Unit tests for PruningEnginePro.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.pruning_engine_pro import PruningEnginePro


class TestPruningEnginePro:
    def test_should_prune_dynamic_bounds(self):
        g1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        g2 = ArcGrid(pixels=np.array([[1, 9], [3, 4]]), background=0)

        pro = PruningEnginePro()
        # Initialize best solution cost to 2
        pro.update_best_cost(2)

        # Path cost 1 + estimated cost-to-go 1 = 2 (exceeds or equal to best cost 2) -> Should prune
        assert pro.should_prune(g1, g2, current_path_cost=1) is True
        # Path cost 0 + estimated cost-to-go 1 = 1 (less than best cost 2) -> Should not prune
        assert pro.should_prune(g1, g2, current_path_cost=0) is False
