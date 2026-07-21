"""
Advanced Vision Features — Symmetry analysis (horizontal, vertical, diagonal, rotational),
2D spatial periodicity detection, and color palette transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid


@dataclass
class SymmetryReport:
    horizontal_score: float
    vertical_score: float
    main_diagonal_score: float
    anti_diagonal_score: float
    rotational_90_score: float

    @property
    def max_symmetry_score(self) -> float:
        return max(
            self.horizontal_score,
            self.vertical_score,
            self.main_diagonal_score,
            self.anti_diagonal_score,
            self.rotational_90_score,
        )


class SymmetryAnalyzer:
    """Analyzes 2D grid symmetry across 5 canonical axes."""

    @staticmethod
    def analyze(grid: ArcGrid) -> SymmetryReport:
        arr = grid.pixels
        h, w = arr.shape

        # Horizontal (left-right flip)
        h_flip = np.fliplr(arr)
        h_score = float(np.mean(arr == h_flip))

        # Vertical (top-bottom flip)
        v_flip = np.flipud(arr)
        v_score = float(np.mean(arr == v_flip))

        # Main diagonal (transpose)
        if h == w:
            d1_score = float(np.mean(arr == arr.T))
            d2_score = float(np.mean(arr == np.rot90(np.fliplr(arr), k=1)))
            rot_score = float(np.mean(arr == np.rot90(arr, k=-1)))
        else:
            d1_score = 0.0
            d2_score = 0.0
            rot_score = 0.0

        return SymmetryReport(
            horizontal_score=h_score,
            vertical_score=v_score,
            main_diagonal_score=d1_score,
            anti_diagonal_score=d2_score,
            rotational_90_score=rot_score,
        )


class PeriodicityDetector:
    """Detects 2D spatial pattern repetition periods (Th, Tw)."""

    @staticmethod
    def detect_period(grid: ArcGrid) -> Tuple[int, int]:
        arr = grid.pixels
        h, w = arr.shape

        period_h = h
        period_w = w

        # Check vertical shift periodicity
        for step in range(1, h // 2 + 1):
            if h % step == 0:
                is_period = True
                for i in range(step, h):
                    if not (arr[i] == arr[i % step]).all():
                        is_period = False
                        break
                if is_period:
                    period_h = step
                    break

        # Check horizontal shift periodicity
        for step in range(1, w // 2 + 1):
            if w % step == 0:
                is_period = True
                for j in range(step, w):
                    if not (arr[:, j] == arr[:, j % step]).all():
                        is_period = False
                        break
                if is_period:
                    period_w = step
                    break

        return period_h, period_w


class ColorPaletteTransfer:
    """Transfers color mapping from source palette to target palette."""

    @staticmethod
    def transfer(source: ArcGrid, target_template: ArcGrid) -> ArcGrid:
        src_colors = [c for c in source.colors if c != source.background]
        tgt_colors = [c for c in target_template.colors if c != target_template.background]

        if not src_colors or not tgt_colors:
            return source

        res = source.pixels.copy()
        for i, sc in enumerate(src_colors):
            tc = tgt_colors[i % len(tgt_colors)]
            res[source.pixels == sc] = tc

        return ArcGrid(pixels=res, background=source.background)
