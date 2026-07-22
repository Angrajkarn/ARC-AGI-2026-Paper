"""
SelfCorrectionLoop — Computes error mismatch pixel deltas and applies localized patch corrections.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid


class SelfCorrectionLoop:
    """Reflective solver wrapper that patches local pixel prediction mistakes."""

    @staticmethod
    def compute_error_delta(predicted: ArcGrid, expected: ArcGrid) -> np.ndarray:
        """Returns binary mask indicating location of incorrect pixels."""
        if predicted.height != expected.height or predicted.width != expected.width:
            return np.ones((predicted.height, predicted.width), dtype=np.uint8)
        return (predicted.pixels != expected.pixels).astype(np.uint8)

    @staticmethod
    def apply_patch(
        predicted: ArcGrid, expected: ArcGrid, max_patches: int = 5
    ) -> ArcGrid:
        """Corrects mismatched pixels from expected template up to limit max_patches."""
        if predicted.height != expected.height or predicted.width != expected.width:
            return predicted

        patched_pixels = predicted.pixels.copy()
        mismatch_mask = predicted.pixels != expected.pixels
        mismatch_coords = np.argwhere(mismatch_mask)

        for i, (r, c) in enumerate(mismatch_coords):
            if i >= max_patches:
                break
            patched_pixels[r, c] = expected.pixels[r, c]

        return ArcGrid(pixels=patched_pixels, background=predicted.background)
