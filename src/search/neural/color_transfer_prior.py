"""
ColorTransferPrior — Translates grid pixels using frequency-ranked color mapping.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.core.grid.grid import ArcGrid


class ColorTransferPrior:
    """Computes color transformations using rank frequency distribution matches."""

    @staticmethod
    def discover_color_mapping(src: ArcGrid, dst: ArcGrid) -> Dict[int, int]:
        """Maps source grid colors to target grid colors by frequency rank (descending)."""
        src_vals, src_counts = np.unique(src.pixels, return_counts=True)
        dst_vals, dst_counts = np.unique(dst.pixels, return_counts=True)

        # Exclude background (0) from frequency ranking if present
        src_bg_idx = np.where(src_vals == src.background)[0]
        if src_bg_idx.size > 0:
            src_vals = np.delete(src_vals, src_bg_idx[0])
            src_counts = np.delete(src_counts, src_bg_idx[0])

        dst_bg_idx = np.where(dst_vals == dst.background)[0]
        if dst_bg_idx.size > 0:
            dst_vals = np.delete(dst_vals, dst_bg_idx[0])
            dst_counts = np.delete(dst_counts, dst_bg_idx[0])

        # Sort values by count descending
        src_sorted = src_vals[np.argsort(-src_counts)]
        dst_sorted = dst_vals[np.argsort(-dst_counts)]

        # Map ranks
        mapping: Dict[int, int] = {}
        for i in range(min(len(src_sorted), len(dst_sorted))):
            mapping[int(src_sorted[i])] = int(dst_sorted[i])

        return mapping
