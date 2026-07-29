"""
SpatialBispectralDensityEstimator — Computes 2D higher-order bispectral phase coupling density.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class SpatialBispectralDensityEstimator:
    """Estimates higher-order 2D bispectral density and non-linear phase coupling."""

    @staticmethod
    def compute_bispectral_density(grid1: ArcGrid, grid2: ArcGrid) -> float:
        """Returns mean 2D Fourier bispectral density B(w1, w2) = F(w1)*F(w2)*conj(F(w1+w2))."""
        if grid1.pixels.shape != grid2.pixels.shape:
            return 0.0

        fft1 = np.fft.fft2(grid1.pixels.astype(float))
        fft2 = np.fft.fft2(grid2.pixels.astype(float))

        # Bispectral density coupling magnitude
        bispectrum = np.abs(fft1 * fft2 * np.conj(fft1 + fft2))
        norm = float(grid1.pixels.size ** 2)

        if norm == 0.0:
            return 0.0

        return float(np.mean(bispectrum)) / norm
