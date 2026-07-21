"""
Neural/LLM-Guided Search & Multi-Dimensional Heuristic — Advanced continuous similarity
and primitive prior probability scoring for MCTS / Beam search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid
from src.core.objects.detector import ObjectDetector
from src.dsl.primitives.transforms import PRIMITIVE_REGISTRY


@dataclass
class HeuristicBreakdown:
    total_score: float
    iou_score: float
    entropy_similarity: float
    color_distribution_score: float
    dimension_score: float


class MultiDimensionalHeuristic:
    """Advanced composite similarity evaluator using IoU, structural entropy, and color alignment."""

    def __init__(self) -> None:
        self.detector = ObjectDetector()

    def _compute_entropy(self, grid: ArcGrid) -> float:
        """Computes normalized Shannon entropy of pixel color distribution."""
        pixels = grid.pixels.flatten()
        if len(pixels) == 0:
            return 0.0
        _, counts = np.unique(pixels, return_counts=True)
        probs = counts / len(pixels)
        ent = -np.sum(probs * np.log2(probs + 1e-9))
        max_ent = np.log2(10.0)
        return float(ent / max_ent)

    def _compute_iou(self, predicted: ArcGrid, expected: ArcGrid) -> float:
        """Computes non-background pixel Intersection-over-Union (IoU)."""
        if (predicted.height, predicted.width) != (expected.height, expected.width):
            return 0.0
        p_mask = (predicted.pixels != predicted.background)
        e_mask = (expected.pixels != expected.background)

        intersection = np.logical_and(p_mask, e_mask).sum()
        union = np.logical_or(p_mask, e_mask).sum()

        if union == 0:
            return 1.0 if (predicted.pixels == expected.pixels).all() else 0.0
        return float(intersection / union)

    def evaluate(self, predicted: ArcGrid, expected: ArcGrid) -> HeuristicBreakdown:
        if predicted == expected:
            return HeuristicBreakdown(
                total_score=1.0,
                iou_score=1.0,
                entropy_similarity=1.0,
                color_distribution_score=1.0,
                dimension_score=1.0,
            )

        # Dimension score
        h_ratio = min(predicted.height, expected.height) / max(predicted.height, expected.height)
        w_ratio = min(predicted.width, expected.width) / max(predicted.width, expected.width)
        dim_score = (h_ratio + w_ratio) / 2.0

        # Entropy score
        p_ent = self._compute_entropy(predicted)
        e_ent = self._compute_entropy(expected)
        ent_score = 1.0 - abs(p_ent - e_ent)

        # Color distribution score
        p_hist = np.bincount(predicted.pixels.flatten(), minlength=10)
        e_hist = np.bincount(expected.pixels.flatten(), minlength=10)
        p_norm = p_hist / (p_hist.sum() + 1e-9)
        e_norm = e_hist / (e_hist.sum() + 1e-9)
        color_score = float(np.sum(np.minimum(p_norm, e_norm)))

        # IoU score
        iou = self._compute_iou(predicted, expected)

        # Composite weighted sum
        total = 0.4 * iou + 0.3 * color_score + 0.15 * ent_score + 0.15 * dim_score
        return HeuristicBreakdown(
            total_score=float(total),
            iou_score=float(iou),
            entropy_similarity=float(ent_score),
            color_distribution_score=float(color_score),
            dimension_score=float(dim_score),
        )


class NeuralSearchPrior:
    """Predicts prior probability distributions over primitive operations given input/output pairs."""

    def __init__(self) -> None:
        self.primitive_names = sorted(list(PRIMITIVE_REGISTRY.keys()))

    def predict_priors(self, input_grid: ArcGrid, target_grid: ArcGrid) -> Dict[str, float]:
        """Returns normalized prior probabilities over available primitive operations."""
        priors: Dict[str, float] = {name: 1.0 for name in self.primitive_names}

        # Spatial transformation heuristics
        if (input_grid.height, input_grid.width) == (target_grid.height, target_grid.width):
            # Same shape encourages rotations, recoloring, reflection
            for p in ["rotate_90", "rotate_180", "mirror_horizontal", "mirror_vertical", "replace_color"]:
                if p in priors:
                    priors[p] += 2.0

        elif input_grid.height * 2 == target_grid.height or input_grid.width * 2 == target_grid.width:
            # Scale heuristics
            for p in ["scale_2x", "pattern_repeat"]:
                if p in priors:
                    priors[p] += 3.0

        # Normalize to probability distribution
        total = sum(priors.values())
        return {k: v / total for k, v in priors.items()}
