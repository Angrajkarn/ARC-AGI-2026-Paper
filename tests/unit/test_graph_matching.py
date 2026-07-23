"""
Unit tests for GraphMatcher.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.core.graphs.scene_graph import SceneGraph
from src.core.objects.detector import ObjectDetector
from src.search.neural.graph_matching import GraphMatcher


class TestGraphMatcher:
    def test_align_graphs(self):
        # Grid A: color 3 pixel at (0,0)
        p_a = np.zeros((3, 3), dtype=np.uint8)
        p_a[0, 0] = 3
        g_a = ArcGrid(pixels=p_a, background=0)

        # Grid B: color 3 pixel at (2,2)
        p_b = np.zeros((3, 3), dtype=np.uint8)
        p_b[2, 2] = 3
        g_b = ArcGrid(pixels=p_b, background=0)

        detector = ObjectDetector()
        graph_a = SceneGraph.build_from_objects(detector.detect(g_a), 3, 3)
        graph_b = SceneGraph.build_from_objects(detector.detect(g_b), 3, 3)

        alignment = GraphMatcher.align_graphs(graph_a, graph_b)
        assert len(alignment) == 1
