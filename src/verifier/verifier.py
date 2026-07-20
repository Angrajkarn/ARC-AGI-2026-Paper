"""
Verifier — evaluate candidate programs against expected outputs.

Verification levels:
  1. Exact match (pixel-perfect)
  2. Object consistency (same object count and colours)
  3. Shape consistency (same bounding boxes)
  4. Colour consistency (same colour histogram)

A program passes only if it achieves exact match on ALL training pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid
from src.core.objects.detector import ObjectDetector
from src.dsl.executor.executor import DSLExecutor
from src.dsl.parser.dsl_parser import DSLProgram
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class VerificationResult:
    """Result of verifying a program against training pairs.

    Attributes:
        passed:          True iff ALL pairs pass exact-match.
        score:           Fraction of pairs that pass (0.0–1.0).
        num_correct:     Number of pairs with exact match.
        num_pairs:       Total number of pairs.
        mismatch_pixels: List of pixel-count mismatches per pair.
        mismatch_sizes:  List of size mismatches (expected vs actual).
        object_scores:   Object-level consistency per pair.
        confidence:      Composite confidence score [0, 1].
        details:         Free-form dict with per-pair diagnostics.
    """

    passed: bool = False
    score: float = 0.0
    num_correct: int = 0
    num_pairs: int = 0
    mismatch_pixels: List[int] = field(default_factory=list)
    mismatch_sizes: List[Tuple] = field(default_factory=list)
    object_scores: List[float] = field(default_factory=list)
    confidence: float = 0.0
    details: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"VerificationResult(passed={self.passed}, score={self.score:.3f}, "
            f"{self.num_correct}/{self.num_pairs} correct)"
        )


class Verifier:
    """Verify candidate programs against ARC training pairs.

    Usage::

        verifier = Verifier()
        result = verifier.verify(program, pairs)
    """

    def __init__(self, require_exact_match: bool = True) -> None:
        self.require_exact_match = require_exact_match
        self._executor = DSLExecutor(debug=False)
        self._detector = ObjectDetector()

    def verify(self, program: DSLProgram, pairs: List[Dict]) -> VerificationResult:
        """Verify a program on all training pairs.

        Args:
            program: The DSLProgram to evaluate.
            pairs:   List of dicts with "input" (ArcGrid) and "output" (ArcGrid).

        Returns:
            VerificationResult with full diagnostics.
        """
        result = VerificationResult(num_pairs=len(pairs))
        if not pairs:
            return result

        pair_details = []
        for i, pair in enumerate(pairs):
            inp: ArcGrid = pair["input"]
            expected: ArcGrid = pair["output"]

            prediction, trace = self._executor.execute(program, inp)

            if prediction is None:
                # Execution failed
                detail = {
                    "pair": i,
                    "exact_match": False,
                    "execution_error": trace.error,
                    "mismatch_pixels": None,
                }
                result.mismatch_pixels.append(-1)
                result.mismatch_sizes.append((expected.size, None))
                result.object_scores.append(0.0)
                pair_details.append(detail)
                continue

            exact = prediction == expected
            size_match = prediction.size == expected.size
            if exact:
                n_mismatch = 0
            elif size_match:
                n_mismatch = int(np.sum(prediction.pixels != expected.pixels))
            else:
                n_mismatch = abs(prediction.num_pixels - expected.num_pixels) + 10
            obj_score = self._object_consistency(prediction, expected) if size_match else 0.0

            detail = {
                "pair": i,
                "exact_match": exact,
                "size_match": size_match,
                "mismatch_pixels": n_mismatch,
                "object_consistency": obj_score,
            }
            pair_details.append(detail)
            result.mismatch_pixels.append(n_mismatch)
            result.mismatch_sizes.append((expected.size, prediction.size))
            result.object_scores.append(obj_score)

            if exact:
                result.num_correct += 1

        result.score = result.num_correct / max(len(pairs), 1)
        result.passed = (result.num_correct == len(pairs)) if self.require_exact_match else (result.score > 0)
        result.confidence = self._compute_confidence(result)
        result.details = {"per_pair": pair_details}

        logger.debug(
            f"Verifier: {result.num_correct}/{len(pairs)} correct, "
            f"score={result.score:.3f}, conf={result.confidence:.3f}"
        )
        return result

    def _object_consistency(self, predicted: ArcGrid, expected: ArcGrid) -> float:
        """Score object-level consistency between predicted and expected grids."""
        try:
            pred_objs = self._detector.detect(predicted)
            exp_objs = self._detector.detect(expected)
        except Exception:
            return 0.0

        if len(pred_objs) != len(exp_objs):
            return 0.5 * (1.0 - abs(len(pred_objs) - len(exp_objs)) / max(len(exp_objs), 1))

        pred_colors = sorted(o.color for o in pred_objs)
        exp_colors = sorted(o.color for o in exp_objs)
        color_match = pred_colors == exp_colors
        return 1.0 if color_match else 0.7

    def _compute_confidence(self, result: VerificationResult) -> float:
        """Compute a composite confidence score."""
        exact_weight = 0.6
        obj_weight = 0.3
        pixel_weight = 0.1

        exact_score = result.score
        obj_score = (
            sum(result.object_scores) / max(len(result.object_scores), 1)
            if result.object_scores else 0.0
        )
        # Pixel proximity: fewer mismatches → higher score
        max_px = max((abs(m) for m in result.mismatch_pixels if m >= 0), default=0) or 1
        px_scores = [
            max(0.0, 1.0 - m / max_px) if m >= 0 else 0.0
            for m in result.mismatch_pixels
        ]
        px_score = sum(px_scores) / max(len(px_scores), 1)

        return (
            exact_weight * exact_score
            + obj_weight * obj_score
            + pixel_weight * px_score
        )


__all__ = ["Verifier", "VerificationResult"]
