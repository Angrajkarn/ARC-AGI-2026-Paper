"""
EquivarianceEncoder — Computes rotation and reflection invariant representation signatures for grid shapes.
"""

from __future__ import annotations

import hashlib
from typing import List

import numpy as np

from src.core.objects.arc_object import ArcObject


class EquivarianceEncoder:
    """Encodes grid shapes into representations invariant under Dihedral (D4) group rotations/flips."""

    @staticmethod
    def compute_invariant_hash(obj: ArcObject) -> str:
        """Generates a stable MD5 hash representing the canonical orientation of the object shape."""
        # Find raw binary mask grid bounds of the object
        r_min, r_max = obj.bounding_box.row_min, obj.bounding_box.row_max
        c_min, c_max = obj.bounding_box.col_min, obj.bounding_box.col_max

        h, w = r_max - r_min + 1, c_max - c_min + 1
        canvas = np.zeros((h, w), dtype=np.uint8)

        for r, c in obj.pixels:
            canvas[r - r_min, c - c_min] = 1

        # Generate all 8 D4 transformations of this canvas mask
        transformations: List[bytes] = []
        for rot in range(4):
            r_canvas = np.rot90(canvas, k=rot)
            transformations.append(r_canvas.tobytes())
            transformations.append(np.fliplr(r_canvas).tobytes())
            transformations.append(np.flipud(r_canvas).tobytes())

        # Sort transformed byte representations lexicographically to choose the canonical version
        transformations.sort()
        canonical_bytes = transformations[0]

        return hashlib.md5(canonical_bytes).hexdigest()
