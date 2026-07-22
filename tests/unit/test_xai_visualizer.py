"""
Unit tests for XAIProgramVisualizer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analytics.xai_visualizer import XAIProgramVisualizer
from src.core.grid.grid import ArcGrid


class TestXAIProgramVisualizer:
    def test_compute_heatmap_same_size(self):
        g1 = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        g2 = ArcGrid(pixels=np.array([[1, 5], [3, 4]]), background=0)

        heatmap = XAIProgramVisualizer.compute_heatmap(g1, g2)
        assert (heatmap == np.array([[0, 1], [0, 0]])).all()

    def test_compute_heatmap_diff_size(self):
        g1 = ArcGrid(pixels=np.array([[1, 2]]), background=0)
        g2 = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)

        heatmap = XAIProgramVisualizer.compute_heatmap(g1, g2)
        assert heatmap.shape == (2, 2)
        assert (heatmap == 1).all()
