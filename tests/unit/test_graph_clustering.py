"""
Unit tests for GraphClustering.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.core.graphs.scene_graph import SceneGraph
from src.core.objects.detector import ObjectDetector
from src.core.graphs.graph_clustering import GraphClustering


class TestGraphClustering:
    def test_cluster_objects(self):
        # A simple grid containing 2 small objects and 1 large object
        pixels = np.zeros((6, 6), dtype=np.uint8)
        pixels[0, 0] = 1  # area 1
        pixels[2, 2] = 2  # area 1
        pixels[4, 4:6] = 3  # area 2
        grid = ArcGrid(pixels=pixels, background=0)

        detector = ObjectDetector()
        graph = SceneGraph.build_from_objects(detector.detect(grid), 6, 6)

        clusters = GraphClustering.cluster_objects(graph)

        # There should be two clusters: one for area 1, one for area 2
        assert len(clusters) == 2
        # Cluster of size 1 (which will be keyed at 1) should contain 2 object indices
        assert len(clusters[1]) == 2
