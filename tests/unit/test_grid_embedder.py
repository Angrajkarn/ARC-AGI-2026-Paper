"""
Unit tests for GridEmbedder.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.grid.grid import ArcGrid
from src.search.neural.grid_embedder import GridEmbedder


class TestGridEmbedder:
    def test_embed_grid_shape(self):
        grid = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
        embedder = GridEmbedder(embedding_dim=16)

        vec = embedder.embed_grid(grid)
        assert vec.shape == (16,)
        assert vec[0] == pytest.approx(2 / 30.0)

    def test_embed_transition(self):
        g1 = ArcGrid(pixels=np.array([[1, 1]]), background=0)
        g2 = ArcGrid(pixels=np.array([[1, 2]]), background=0)

        embedder = GridEmbedder(embedding_dim=8)
        trans_vec = embedder.embed_transition(g1, g2)

        assert trans_vec.shape == (8,)
