"""
ScalePrior — Computes input-output grid dimension scaling priors.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid


class ScalePrior:
    """Predicts output grid dimensions based on input dimensions and training examples."""

    @staticmethod
    def estimate_scale_factor(examples: List[Tuple[ArcGrid, ArcGrid]]) -> Tuple[float, float]:
        """Calculates mean height scale factor and mean width scale factor."""
        if not examples:
            return 1.0, 1.0

        h_ratios = []
        w_ratios = []

        for inp, out in examples:
            h_in, w_in = inp.pixels.shape
            h_out, w_out = out.pixels.shape
            h_ratios.append(float(h_out) / float(h_in))
            w_ratios.append(float(w_out) / float(w_in))

        return float(np.mean(h_ratios)), float(np.mean(w_ratios))
