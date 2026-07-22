"""
Unit tests for GNNRelationEncoder.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.core.graphs.scene_graph import SceneGraph
from src.core.objects.detector import ObjectDetector
from src.search.neural.gnn_relation import GNNRelationEncoder


class TestGNNRelation:
    def test_encode_relations(self):
        # Two separated colored pixels in 3x3 grid forming nodes in scene graph
        pixels = np.zeros((3, 3), dtype=np.uint8)
        pixels[0, 0] = 1
        pixels[2, 2] = 2
        grid = ArcGrid(pixels=pixels, background=0)

        detector = ObjectDetector()
        objects = detector.detect(grid)

        graph = SceneGraph.build_from_objects(objects, grid.height, grid.width)
        encoder = GNNRelationEncoder(passes=1)

        embeddings = encoder.encode_relations(graph)
        # Ensure embedding dimensions are valid
        assert len(embeddings) >= 2
        for emb in embeddings.values():
            assert emb.shape == (4,)
