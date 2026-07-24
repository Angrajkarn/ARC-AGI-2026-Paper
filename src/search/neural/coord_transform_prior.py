"""
CoordTransformPrior — Validates coordinate grid transformations matching D4 group actions.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class CoordTransformPrior:
    """Verifies coordinate shifts under reflections and rotations."""

    @staticmethod
    def transform_coordinates(
        coords: List[Tuple[int, int]],
        grid_height: int,
        grid_width: int,
        action: str
    ) -> List[Tuple[int, int]]:
        """Maps coordinates according to reflection or rotation actions."""
        transformed = []
        for r, c in coords:
            if action == "mirror_horizontal":
                transformed.append((r, grid_width - 1 - c))
            elif action == "mirror_vertical":
                transformed.append((grid_height - 1 - r, c))
            elif action == "rotate_180":
                transformed.append((grid_height - 1 - r, grid_width - 1 - c))
            else:
                # Identity fallback
                transformed.append((r, c))

        return transformed
