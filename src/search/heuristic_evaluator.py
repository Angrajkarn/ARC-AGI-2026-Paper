"""
Smooth Gradient Heuristic Evaluator — continuous similarity scoring for search engines.

Computes a composite similarity score in [0.0, 1.0] between predicted and expected ArcGrid:
  - Pixel Hamming Similarity
  - Color Histogram Cosine Similarity
  - Bounding Box & Dimension Similarity
  - Object Count & Area Similarity
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid
from src.core.objects.detector import ObjectDetector
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SimilarityScore:
    total_score: float
    pixel_similarity: float
    color_similarity: float
    dimension_similarity: float
    object_similarity: float


class HeuristicEvaluator:
    """Compute continuous gradient similarity scores between ArcGrid pairs."""

    def __init__(self) -> None:
        self._detector = ObjectDetector()

    def evaluate_similarity(self, predicted: ArcGrid, expected: ArcGrid) -> SimilarityScore:
        """Compute continuous similarity score in [0.0, 1.0] between predicted and expected grids.

        Args:
            predicted: Candidate grid output.
            expected: Target ground truth grid.

        Returns:
            SimilarityScore instance.
        """
        # Exact match check
        if predicted == expected:
            return SimilarityScore(
                total_score=1.0,
                pixel_similarity=1.0,
                color_similarity=1.0,
                dimension_similarity=1.0,
                object_similarity=1.0,
            )

        # 1. Dimension similarity
        dim_sim = (
            min(predicted.height, expected.height) / max(predicted.height, expected.height)
        ) * (
            min(predicted.width, expected.width) / max(predicted.width, expected.width)
        )

        # 2. Pixel similarity (only if dimensions match exactly)
        if predicted.size == expected.size:
            matches = int(np.sum(predicted.pixels == expected.pixels))
            px_sim = matches / max(expected.num_pixels, 1)
        else:
            px_sim = 0.0

        # 3. Color histogram cosine similarity
        color_sim = self._color_histogram_similarity(predicted, expected)

        # 4. Object similarity
        obj_sim = self._object_count_similarity(predicted, expected)

        # Weighted composite score
        total = (
            0.50 * px_sim
            + 0.20 * color_sim
            + 0.15 * dim_sim
            + 0.15 * obj_sim
        )

        return SimilarityScore(
            total_score=round(total, 4),
            pixel_similarity=round(px_sim, 4),
            color_similarity=round(color_sim, 4),
            dimension_similarity=round(dim_sim, 4),
            object_similarity=round(obj_sim, 4),
        )

    def _color_histogram_similarity(self, a: ArcGrid, b: ArcGrid) -> float:
        """Compute cosine similarity of 10-bin color histograms."""
        hist_a = np.bincount(a.pixels.flatten(), minlength=10).astype(float)
        hist_b = np.bincount(b.pixels.flatten(), minlength=10).astype(float)

        norm_a = np.linalg.norm(hist_a)
        norm_b = np.linalg.norm(hist_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        cosine_sim = float(np.dot(hist_a, hist_b) / (norm_a * norm_b))
        return max(0.0, min(1.0, cosine_sim))

    def _object_count_similarity(self, a: ArcGrid, b: ArcGrid) -> float:
        """Compare object counts between two grids."""
        try:
            objs_a = len(self._detector.detect(a))
            objs_b = len(self._detector.detect(b))
            if objs_a == objs_b:
                return 1.0
            return max(0.0, 1.0 - abs(objs_a - objs_b) / max(objs_b, 1))
        except Exception:
            return 0.5


__all__ = ["HeuristicEvaluator", "SimilarityScore"]
