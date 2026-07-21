"""
FailureAnalyzer — Computes task entropy, object complexity metrics, and failure clusters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.core.grid.grid import ArcGrid
from src.core.objects.detector import ObjectDetector


@dataclass
class TaskComplexityMetrics:
    shannon_entropy: float
    num_objects: int
    grid_area: int
    unique_colors: int


class FailureAnalyzer:
    """Analyzes benchmark execution logs and categorizes task difficulties."""

    def __init__(self) -> None:
        self.detector = ObjectDetector()

    def compute_complexity(self, grid: ArcGrid) -> TaskComplexityMetrics:
        """Computes complexity metrics for a single ArcGrid."""
        # Shannon Entropy over color histogram
        counts = np.bincount(grid.pixels.ravel(), minlength=10)
        probs = counts / counts.sum()
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        objects = self.detector.detect(grid)
        unique_colors = len(grid.colors)
        area = grid.height * grid.width

        return TaskComplexityMetrics(
            shannon_entropy=round(entropy, 4),
            num_objects=len(objects),
            grid_area=area,
            unique_colors=unique_colors,
        )

    def generate_report(self, task_grids: Dict[str, ArcGrid]) -> Dict[str, TaskComplexityMetrics]:
        """Generates complexity breakdown for a dictionary of task input grids."""
        return {task_id: self.compute_complexity(grid) for task_id, grid in task_grids.items()}
