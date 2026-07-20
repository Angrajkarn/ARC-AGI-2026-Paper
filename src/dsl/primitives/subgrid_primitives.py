"""
Subgrid & Tile Extraction Primitives — pattern tile extraction, tiling, and grid line partitioning.

Primitives:
  - extract_repeating_tile(grid)
  - tile_to_fit(grid, target_height, target_width)
  - split_by_grid_lines(grid, line_color)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid
from src.dsl.primitives.transforms import PRIMITIVE_REGISTRY


def extract_repeating_tile(grid: ArcGrid) -> ArcGrid:
    """Find the smallest repeating 2D block in the grid."""
    H, W = grid.height, grid.width
    arr = grid.pixels

    for h in range(1, H + 1):
        if H % h != 0:
            continue
        for w in range(1, W + 1):
            if W % w != 0:
                continue
            if h == H and w == W:
                continue

            tile = arr[:h, :w]
            tiled = np.tile(tile, (H // h, W // w))
            if np.array_equal(arr, tiled):
                return ArcGrid(pixels=tile.copy(), background=grid.background)

    return grid.copy()


def tile_to_fit(grid: ArcGrid, target_height: int = 6, target_width: int = 6) -> ArcGrid:
    """Repeat grid to fill a canvas of size (target_height, target_width)."""
    h_reps = (target_height + grid.height - 1) // max(grid.height, 1)
    w_reps = (target_width + grid.width - 1) // max(grid.width, 1)

    tiled = np.tile(grid.pixels, (h_reps, w_reps))
    cropped = tiled[:target_height, :target_width]
    return ArcGrid(pixels=cropped.copy(), background=grid.background)


def split_by_grid_lines(grid: ArcGrid, line_color: int = 5) -> List[ArcGrid]:
    """Extract cell subgrids partitioned by horizontal and vertical dividing lines of line_color."""
    arr = grid.pixels
    H, W = arr.shape

    # Find row indices that are full grid lines of line_color
    row_mask = (arr == line_color).all(axis=1)
    col_mask = (arr == line_color).all(axis=0)

    row_splits = [i for i, val in enumerate(row_mask) if val]
    col_splits = [j for j, val in enumerate(col_mask) if val]

    row_bounds = [0] + row_splits + [H]
    col_bounds = [0] + col_splits + [W]

    cells: List[ArcGrid] = []
    for r in range(len(row_bounds) - 1):
        r1, r2 = row_bounds[r], row_bounds[r + 1]
        if r1 in row_splits:
            r1 += 1
        if r1 >= r2:
            continue
        for c in range(len(col_bounds) - 1):
            c1, c2 = col_bounds[c], col_bounds[c + 1]
            if c1 in col_splits:
                c1 += 1
            if c1 >= c2:
                continue
            cell_arr = arr[r1:r2, c1:c2]
            cells.append(ArcGrid(pixels=cell_arr.copy(), background=grid.background))

    return cells or [grid.copy()]


# Register into PRIMITIVE_REGISTRY
PRIMITIVE_REGISTRY.update({
    "extract_repeating_tile": extract_repeating_tile,
    "tile_to_fit": tile_to_fit,
})

__all__ = ["extract_repeating_tile", "tile_to_fit", "split_by_grid_lines"]
