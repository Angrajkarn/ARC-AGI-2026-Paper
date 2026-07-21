"""
Unit tests for FailureAnalyzer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analytics.failure_analyzer import FailureAnalyzer, TaskComplexityMetrics
from src.core.grid.grid import ArcGrid


class TestFailureAnalyzer:
    def test_compute_complexity(self):
        grid = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        analyzer = FailureAnalyzer()

        metrics = analyzer.compute_complexity(grid)
        assert isinstance(metrics, TaskComplexityMetrics)
        assert metrics.grid_area == 4
        assert metrics.unique_colors == 4

    def test_generate_report(self):
        g1 = ArcGrid(pixels=np.array([[1, 1], [1, 1]]), background=0)
        analyzer = FailureAnalyzer()

        report = analyzer.generate_report({"t1": g1})
        assert "t1" in report
        assert report["t1"].unique_colors == 1
