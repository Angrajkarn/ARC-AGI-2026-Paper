"""
SkeletonPersistenceLandscapeMatcher — Computes persistence landscape vector profiles on skeletal graphs.
"""

from __future__ import annotations

import math
from typing import List, Set, Tuple

import numpy as np


class SkeletonPersistenceLandscapeMatcher:
    """Calculates persistence landscape vector summaries for topological skeletal structures."""

    @staticmethod
    def compute_landscape_vector(skeleton_pixels: Set[Tuple[int, int]], num_bins: int = 5) -> List[float]:
        """Returns 1D landscape feature summary vector (binned spatial densities from centroid)."""
        if not skeleton_pixels:
            return [0.0] * num_bins

        # Compute centroid
        rows = [r for r, c in skeleton_pixels]
        cols = [c for r, c in skeleton_pixels]
        mean_r = sum(rows) / float(len(rows))
        mean_c = sum(cols) / float(len(cols))

        # Radial distances from centroid
        dists = [math.sqrt((r - mean_r) ** 2 + (c - mean_c) ** 2) for r, c in skeleton_pixels]
        max_d = max(dists) if max(dists) > 0 else 1.0

        hist, _ = np.histogram(dists, bins=num_bins, range=(0.0, max_d))
        norm_hist = (hist.astype(float) / float(len(skeleton_pixels))).tolist()
        return norm_hist
