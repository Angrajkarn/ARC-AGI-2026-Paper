"""
GridEmbedder — Computes high-dimensional spatial transition vectors representing transformation types.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.core.grid.grid import ArcGrid


class GridEmbedder:
    """Encodes grid spatial structures and input-output transitions into embedding vectors."""

    def __init__(self, embedding_dim: int = 16) -> None:
        self.embedding_dim = embedding_dim

    def embed_grid(self, grid: ArcGrid) -> np.ndarray:
        """Generates static grid shape and color embedding vector."""
        vec = np.zeros(self.embedding_dim)
        # Seed features: height, width, unique color ratio
        vec[0] = grid.height / 30.0
        vec[1] = grid.width / 30.0
        vec[2] = len(grid.colors) / 10.0

        # Pixel density and background ratio
        non_bg_ratio = (grid.pixels != grid.background).sum() / grid.pixels.size
        vec[3] = non_bg_ratio

        # Fill remaining elements with deterministically seeded noise derived from pixel content
        pixels_hash = hash(grid.pixels.tobytes())
        np.random.seed(abs(pixels_hash) % (2**32))
        vec[4:] = np.random.uniform(-1, 1, self.embedding_dim - 4)

        return vec

    def embed_transition(self, input_grid: ArcGrid, output_grid: ArcGrid) -> np.ndarray:
        """Computes transition embedding vector representing input-to-output changes."""
        in_embed = self.embed_grid(input_grid)
        out_embed = self.embed_grid(output_grid)
        return out_embed - in_embed
