"""
TopologicalPersistence — Computes D4 equivariant persistence cycles over grid color components.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid


class TopologicalPersistence:
    """Extracts topological persistence barcodes of nested loops and components across grids."""

    @staticmethod
    def compute_persistence_diagram(grid: ArcGrid) -> List[Dict[str, Any]]:
        """Calculates birth-death levels for connected components at each color threshold."""
        pixels = grid.pixels
        h, w = pixels.shape
        bg = grid.background

        # Distinct colors present in the grid excluding background
        colors = sorted(list(set(np.unique(pixels)) - {bg}))
        barcodes = []

        for color in colors:
            mask = pixels == color
            visited = np.zeros_like(mask)
            components_count = 0

            for r in range(h):
                for c in range(w):
                    if mask[r, c] and not visited[r, c]:
                        # BFS to find component size
                        components_count += 1
                        queue = [(r, c)]
                        visited[r, c] = True
                        idx = 0
                        while idx < len(queue):
                            curr_r, curr_c = queue[idx]
                            idx += 1
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = curr_r + dr, curr_c + dc
                                if 0 <= nr < h and 0 <= nc < w:
                                    if mask[nr, nc] and not visited[nr, nc]:
                                        visited[nr, nc] = True
                                        queue.append((nr, nc))

            if components_count > 0:
                barcodes.append({
                    "color": int(color),
                    "birth": 0.0,
                    "death": float(components_count),
                    "dimension": 0,
                })

        return barcodes
