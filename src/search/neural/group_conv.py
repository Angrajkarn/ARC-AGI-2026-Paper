"""
GroupConvPrior — D4 symmetry-equivariant feature map estimator.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class GroupConvPrior:
    """Simulates a D4 group equivariant convolutional filter prior over grid patterns."""

    def __init__(self, kernel_size: int = 3) -> None:
        self.kernel_size = kernel_size
        # Seed D4 equivariant symmetric kernel: uniform weights
        self.kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

    def apply_equivariant_conv(self, grid: ArcGrid) -> np.ndarray:
        """Applies convolution to input grid, producing feature map equivariant to D4 symmetries."""
        pixels = grid.pixels.astype(float)
        h, w = pixels.shape

        # Zero pad pixels for convolution boundary matching
        pad = self.kernel_size // 2
        padded = np.pad(pixels, pad, mode="constant", constant_values=grid.background)

        feature_map = np.zeros_like(pixels)
        for r in range(h):
            for c in range(w):
                region = padded[r : r + self.kernel_size, c : c + self.kernel_size]
                feature_map[r, c] = np.sum(region * self.kernel)

        return feature_map
