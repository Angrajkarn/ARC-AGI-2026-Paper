"""
AutonomousScientist — Analyzes failure patterns to suggest DSL expansion hypotheses.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.core.grid.grid import ArcGrid


class AutonomousScientist:
    """Scientific research generator that discovers limitations in search runs and proposes new rules."""

    @staticmethod
    def analyze_failure(predicted: ArcGrid, expected: ArcGrid) -> Dict[str, Any]:
        """Analyzes spatial and color mismatches to formulate rule-improvement hypotheses."""
        if predicted.height != expected.height or predicted.width != expected.width:
            return {
                "failure_mode": "dimensions_mismatch",
                "recommended_fix": "add_resizing_or_cropping_rules",
                "confidence": 0.9,
            }

        # Grid size matches, inspect pixel differences
        mismatched_pixels = predicted.pixels != expected.pixels
        diff_count = np.sum(mismatched_pixels)
        if diff_count == 0:
            return {
                "failure_mode": "none",
                "recommended_fix": "none",
                "confidence": 1.0,
            }

        expected_colors = set(expected.pixels[mismatched_pixels])
        pred_colors = set(predicted.pixels[mismatched_pixels])

        # Color substitution recommendation
        if len(expected_colors) == 1 and len(pred_colors) == 1:
            return {
                "failure_mode": "color_shift",
                "recommended_fix": f"add_color_mapping_{list(pred_colors)[0]}_to_{list(expected_colors)[0]}",
                "confidence": 0.85,
            }

        # Dense region errors
        if diff_count > (predicted.height * predicted.width) * 0.5:
            return {
                "failure_mode": "structural_overhaul",
                "recommended_fix": "add_coordinate_rotation_or_reflection_symmetry",
                "confidence": 0.7,
            }

        # Default fallback
        return {
            "failure_mode": "localized_noise",
            "recommended_fix": "add_relative_translation_or_stencil_masking",
            "confidence": 0.5,
        }
