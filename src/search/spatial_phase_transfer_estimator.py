"""
SpatialPhaseTransferEstimator — Computes 2D spatial frequency transfer functions H(u,v) between grid pairs.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class SpatialPhaseTransferEstimator:
    """Estimates frequency-domain 2D spatial transfer function magnitude and phase response."""

    @staticmethod
    def compute_transfer_function_magnitude(input_grid: ArcGrid, output_grid: ArcGrid) -> float:
        """Returns mean 2D transfer function gain |H(u,v)| = |Y(u,v)| / (|X(u,v)| + eps)."""
        if input_grid.pixels.shape != output_grid.pixels.shape:
            return 0.0

        x_fft = np.fft.fft2(input_grid.pixels.astype(float))
        y_fft = np.fft.fft2(output_grid.pixels.astype(float))

        x_abs = np.abs(x_fft)
        y_abs = np.abs(y_fft)

        eps = 1e-6
        # Where both are zero, transfer gain is 1.0
        mask_both_zero = (x_abs < eps) & (y_abs < eps)
        h_func = np.zeros_like(x_abs)

        valid_mask = ~mask_both_zero
        h_func[mask_both_zero] = 1.0
        h_func[valid_mask] = y_abs[valid_mask] / (x_abs[valid_mask] + eps)

        return float(np.mean(h_func))
