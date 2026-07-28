"""
SpatialSpectralDensityEstimator — Computes 2D cross-spectral power density in frequency domain.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class SpatialSpectralDensityEstimator:
    """Estimates cross-spectral power density between grid candidate outputs and target grids."""

    @staticmethod
    def compute_spectral_density(grid1: ArcGrid, grid2: ArcGrid) -> float:
        """Returns mean 2D Fourier cross-spectral power density value."""
        if grid1.pixels.shape != grid2.pixels.shape:
            return 0.0

        fft1 = np.fft.fft2(grid1.pixels.astype(float))
        fft2 = np.fft.fft2(grid2.pixels.astype(float))

        # Cross-spectral density S_xy = F_x * conj(F_y)
        csd = fft1 * np.conj(fft2)
        power_density = np.abs(csd)

        norm = float(grid1.pixels.size)
        if norm == 0.0:
            return 0.0

        return float(np.mean(power_density)) / norm
