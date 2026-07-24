"""
SubgridAttention — Computes cosine similarity self-attention scores across local 3x3 sub-grids.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class SubgridAttention:
    """Estimates pattern regularity by checking 3x3 local patch matching profiles."""

    @staticmethod
    def compute_self_attention(grid: ArcGrid) -> np.ndarray:
        """Extracts 3x3 patches, flattens them, and returns a patch similarity matrix."""
        pixels = grid.pixels
        h, w = pixels.shape
        patches = []

        # Extract all possible 3x3 patches
        for r in range(h - 2):
            for c in range(w - 2):
                patch = pixels[r:r+3, c:c+3].flatten().astype(float)
                norm = np.linalg.norm(patch)
                if norm > 0:
                    patch = patch / norm
                patches.append(patch)

        if not patches:
            return np.zeros((0, 0))

        patch_matrix = np.array(patches)
        # Compute pairwise dot products
        attention = np.dot(patch_matrix, patch_matrix.T)
        return attention
