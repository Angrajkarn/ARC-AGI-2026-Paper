"""
Output Shape Predictor — predict output grid dimensions for ARC test inputs.

Predicts candidate (height, width) dimensions based on training pair relationships:
  - Fixed output size across all pairs (e.g., 3x3 output)
  - Preserves input size (1:1 ratio)
  - Uniform scaling factor (k * H, k * W)
  - Bounding box of content / largest object
  - Proportional to number of objects
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid
from src.core.objects.detector import ObjectDetector
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ShapePrediction:
    """Predicted output grid dimension candidate."""

    height: int
    width: int
    confidence: float
    rule_type: str  # "fixed", "identity", "scale", "bounding_box", "object_count"


class ShapePredictor:
    """Predict candidate output grid shapes for an input grid based on training pairs."""

    def __init__(self) -> None:
        self._detector = ObjectDetector()

    def predict(
        self, pairs: List[Dict], test_input: ArcGrid
    ) -> List[ShapePrediction]:
        """Predict possible output grid dimensions for test_input.

        Args:
            pairs:      Training pairs with "input" and "output" ArcGrids.
            test_input: The test input ArcGrid.

        Returns:
            Ranked list of ShapePrediction candidates sorted by confidence descending.
        """
        if not pairs:
            return [ShapePrediction(test_input.height, test_input.width, 0.5, "identity")]

        predictions: List[ShapePrediction] = []

        # Rule 1: Fixed output size across all training pairs
        out_sizes = [p["output"].size for p in pairs]
        if len(set(out_sizes)) == 1:
            h, w = out_sizes[0]
            predictions.append(
                ShapePrediction(h, w, 1.0, "fixed")
            )

        # Rule 2: Identity / Preserves size
        is_identity = all(p["input"].size == p["output"].size for p in pairs)
        if is_identity:
            predictions.append(
                ShapePrediction(test_input.height, test_input.width, 0.95, "identity")
            )

        # Rule 3: Uniform integer scaling factor
        scale_factors = []
        for p in pairs:
            ih, iw = p["input"].size
            oh, ow = p["output"].size
            if ih > 0 and iw > 0 and oh % ih == 0 and ow % iw == 0 and (oh // ih) == (ow // iw):
                scale_factors.append(oh // ih)
            else:
                scale_factors.append(None)

        if all(sf is not None and sf == scale_factors[0] for sf in scale_factors):
            k = scale_factors[0]
            if k is not None:
                predictions.append(
                    ShapePrediction(
                        test_input.height * k,
                        test_input.width * k,
                        0.9,
                        f"scale_{k}x",
                    )
                )

        # Rule 4: Content Bounding Box size
        bbox_match = True
        for p in pairs:
            content_h, content_w = self._get_content_bbox_size(p["input"])
            if (content_h, content_w) != p["output"].size:
                bbox_match = False
                break
        if bbox_match:
            th, tw = self._get_content_bbox_size(test_input)
            predictions.append(
                ShapePrediction(th, tw, 0.85, "bounding_box")
            )

        # Fallback: Default to input size if no higher-confidence rule triggered
        if not predictions:
            predictions.append(
                ShapePrediction(test_input.height, test_input.width, 0.5, "identity")
            )

        # Deduplicate predictions by (height, width) keeping max confidence
        unique_preds: Dict[Tuple[int, int], ShapePrediction] = {}
        for p in predictions:
            key = (p.height, p.width)
            if key not in unique_preds or p.confidence > unique_preds[key].confidence:
                unique_preds[key] = p

        result = sorted(unique_preds.values(), key=lambda x: x.confidence, reverse=True)
        logger.debug(f"ShapePredictor: predicted {len(result)} candidates for test grid: {result}")
        return result

    def _get_content_bbox_size(self, grid: ArcGrid) -> Tuple[int, int]:
        """Compute the height and width of non-background bounding box."""
        non_bg_pos = grid.pixels != grid.background
        if not non_bg_pos.any():
            return (grid.height, grid.width)
        rows, cols = np.where(non_bg_pos)
        rmin, rmax = int(rows.min()), int(rows.max())
        cmin, cmax = int(cols.min()), int(cols.max())
        return (rmax - rmin + 1, cmax - cmin + 1)


__all__ = ["ShapePredictor", "ShapePrediction"]
