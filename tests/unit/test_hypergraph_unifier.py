"""
Unit tests for HypergraphUnifier.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.core.graphs.hypergraph_unifier import HypergraphUnifier, HyperEdge


class TestHypergraphUnifier:
    def test_build_hypergraph_color_groups(self):
        # Two separated color 1 squares: should form a color_group hyperedge
        pixels = np.zeros((5, 5), dtype=np.uint8)
        pixels[1, 1] = 1
        pixels[3, 3] = 1
        grid = ArcGrid(pixels=pixels, background=0)

        unifier = HypergraphUnifier()
        edges = unifier.build_hypergraph(grid)

        color_groups = [e for e in edges if e.relation_type == "color_group"]
        assert len(color_groups) >= 1
