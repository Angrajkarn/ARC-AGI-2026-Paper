"""
PruningEnginePro — Advanced A* dynamic cost-bound search branch pruner.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid
from src.search.pruning_engine import PruningEngine


class PruningEnginePro:
    """Manages dynamic cost threshold tracking to prune search nodes exceeding A* bounds."""

    def __init__(self) -> None:
        self.engine = PruningEngine()
        self.best_solution_cost = float("inf")

    def update_best_cost(self, cost: int) -> None:
        """Sets lower cost bound based on the best valid program found so far."""
        if cost < self.best_solution_cost:
            self.best_solution_cost = cost

    def should_prune(self, current: ArcGrid, target: ArcGrid, current_path_cost: int) -> bool:
        """Prunes search branch if path cost + estimated cost-to-go exceeds current best cost."""
        estimated = self.engine.estimate_cost_to_go(current, target)
        total_estimated = current_path_cost + estimated
        return total_estimated >= self.best_solution_cost
