"""
BoundaryAligner — Detects alignment patterns between segmented shape borders.
"""

from __future__ import annotations

from typing import List, Set, Tuple


class BoundaryAligner:
    """Finds if distinct objects align horizontally or vertically."""

    @staticmethod
    def check_alignment(
        obj1: Set[Tuple[int, int]],
        obj2: Set[Tuple[int, int]]
    ) -> Tuple[bool, bool]:
        """Returns (horizontal_aligned, vertical_aligned) if boundaries share row/col indices."""
        if not obj1 or not obj2:
            return False, False

        rows1 = {r for r, c in obj1}
        cols1 = {c for r, c in obj1}

        rows2 = {r for r, c in obj2}
        cols2 = {c for r, c in obj2}

        # Align horizontally if they share at least one row index
        h_align = len(rows1.intersection(rows2)) > 0
        # Align vertically if they share at least one column index
        v_align = len(cols1.intersection(cols2)) > 0

        return h_align, v_align
