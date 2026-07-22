"""
ColorEquivarianceMatcher — Discovers bijective mapping permutations between input and output grid color sets.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.core.grid.grid import ArcGrid


class ColorEquivarianceMatcher:
    """Finds color-mapping permutations to solve color-agnostic tasks."""

    @staticmethod
    def find_mapping(train_pairs: List[Dict[str, ArcGrid]]) -> Optional[Dict[int, int]]:
        """Identifies bijective mapping between non-background input and output colors."""
        if not train_pairs:
            return None

        mapping: Dict[int, int] = {}
        for pair in train_pairs:
            in_grid = pair["input"]
            out_grid = pair["output"]

            if in_grid.height != out_grid.height or in_grid.width != out_grid.width:
                return None

            h, w = in_grid.height, in_grid.width
            for r in range(h):
                for c in range(w):
                    in_val = in_grid.get(r, c)
                    out_val = out_grid.get(r, c)

                    if in_val == in_grid.background:
                        continue

                    # If mapping conflict occurs, bijection fails
                    if in_val in mapping and mapping[in_val] != out_val:
                        return None
                    mapping[in_val] = out_val

        return mapping
