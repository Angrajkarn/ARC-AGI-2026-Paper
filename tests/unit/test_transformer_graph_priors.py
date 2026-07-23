"""
Unit tests for TransformerGraphPriors.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.core.graphs.scene_graph import SceneGraph
from src.core.objects.detector import ObjectDetector
from src.search.neural.transformer_graph_priors import TransformerGraphPriors


class TestTransformerGraphPriors:
    def test_compute_prior_distribution(self):
        # A simple grid with 2 objects
        pixels = np.zeros((4, 4), dtype=np.uint8)
        pixels[0, 0] = 1
        pixels[3, 3] = 2
        grid = ArcGrid(pixels=pixels, background=0)

        detector = ObjectDetector()
        graph = SceneGraph.build_from_objects(detector.detect(grid), 4, 4)

        model = TransformerGraphPriors(embedding_dim=4)
        dist = model.compute_prior_distribution(graph)

        # Output weight distribution vector should match size embedding_dim (4)
        assert dist.shape == (2,)
