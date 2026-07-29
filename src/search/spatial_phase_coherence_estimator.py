"""
SpatialPhaseCoherenceEstimator — Computes 2D Fourier phase coherence across spatial frequency components.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class SpatialPhaseCoherenceEstimator:
    """Estimates phase alignment and coherence across spatial frequency spectrum."""

    @staticmethod
    def compute_phase_coherence(grid1: ArcGrid, grid2: ArcGrid) -> float:
        """Returns mean cosine of 2D Fourier phase angle differences in [0.0, 1.0]."""
        if grid1.pixels.shape != grid2.pixels.shape:
            return 0.0

        fft1 = np.fft.fft2(grid1.pixels.astype(float))
        fft2 = np.fft.fft2(grid2.pixels.astype(float))

        angle1 = np.angle(fft1)
        angle2 = np.angle(fft2)

        # Mean phase difference cosine
        phase_diff = np.cos(angle1 - angle2)
        coherence = float(np.mean(phase_diff))

        # Map [-1.0, 1.0] to [0.0, 1.0]
        norm_coherence = 0.5 * (coherence + 1.0)
        return float(np.clip(norm_coherence, 0.0, 1.0))
