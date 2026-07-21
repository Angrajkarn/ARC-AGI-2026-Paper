"""
InvariantProofVerifier — Verifies mathematical grid invariants (mass conservation, color bounds, dimension constraints).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.core.grid.grid import ArcGrid


@dataclass
class ProofReport:
    is_valid: bool
    mass_conserved: bool
    colors_valid: bool
    dimensions_valid: bool
    reasons: List[str]


class InvariantProofVerifier:
    """Verifies that predicted grids satisfy physical grid invariants."""

    @staticmethod
    def verify(predicted: ArcGrid, expected_template: Optional[ArcGrid] = None) -> ProofReport:
        reasons = []
        mass_conserved = True
        colors_valid = True
        dimensions_valid = True

        # Check color bounds [0..9]
        if (predicted.pixels < 0).any() or (predicted.pixels > 9).any():
            colors_valid = False
            reasons.append("Grid contains out-of-bound color values outside [0..9].")

        # Check dimension limits [1..30]
        if predicted.height < 1 or predicted.height > 30 or predicted.width < 1 or predicted.width > 30:
            dimensions_valid = False
            reasons.append(f"Grid dimensions {predicted.height}x{predicted.width} exceed ARC limits [1..30].")

        # Compare with expected template if provided
        if expected_template is not None:
            p_non_bg = (predicted.pixels != predicted.background).sum()
            e_non_bg = (expected_template.pixels != expected_template.background).sum()

            if abs(p_non_bg - e_non_bg) > (expected_template.pixels.size * 0.5):
                mass_conserved = False
                reasons.append("Pixel mass deviates excessively from target template.")

        is_valid = colors_valid and dimensions_valid and mass_conserved
        return ProofReport(
            is_valid=is_valid,
            mass_conserved=mass_conserved,
            colors_valid=colors_valid,
            dimensions_valid=dimensions_valid,
            reasons=reasons,
        )
