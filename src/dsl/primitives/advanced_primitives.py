"""
Advanced DSL Primitives — Grid morphology, pattern tiling, cellular automata,
enclosed hole filling, and outline extraction.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from src.core.grid.grid import ArcGrid
from src.dsl.primitives.transforms import _register, PRIMITIVE_REGISTRY


@_register("dilate")
def dilate(grid: ArcGrid, color: Optional[int] = None, radius: int = 1) -> ArcGrid:
    """
    Morphological dilation on non-background pixels (or specific color).
    Expands object boundaries by *radius* pixels.
    """
    arr = grid.pixels.copy()
    bg = grid.background

    if color is not None:
        mask = (arr == color)
        target_color = color
    else:
        mask = (arr != bg)
        # Find dominant non-background color
        non_bg_colors = arr[arr != bg]
        target_color = int(non_bg_colors[0]) if len(non_bg_colors) > 0 else 1

    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    dilated_mask = binary_dilation(mask, structure=struct)

    res = arr.copy()
    res[dilated_mask & (arr == bg)] = target_color
    return ArcGrid(pixels=res, background=bg)


@_register("erode")
def erode(grid: ArcGrid, color: Optional[int] = None, radius: int = 1) -> ArcGrid:
    """
    Morphological erosion on non-background pixels (or specific color).
    Shrinks object boundaries by *radius* pixels.
    """
    arr = grid.pixels.copy()
    bg = grid.background

    if color is not None:
        mask = (arr == color)
    else:
        mask = (arr != bg)

    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    eroded_mask = binary_erosion(mask, structure=struct)

    res = arr.copy()
    res[mask & ~eroded_mask] = bg
    return ArcGrid(pixels=res, background=bg)


@_register("extract_outline")
def extract_outline(grid: ArcGrid, color: Optional[int] = None, outline_color: Optional[int] = None) -> ArcGrid:
    """
    Extracts the outer boundary / outline of objects, leaving inner core as background.
    """
    arr = grid.pixels.copy()
    bg = grid.background
    out_col = outline_color if outline_color is not None else 1

    if color is not None:
        mask = (arr == color)
        out_col = outline_color if outline_color is not None else color
    else:
        mask = (arr != bg)

    # Erosion gives interior
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    interior = binary_erosion(mask, structure=struct)
    outline_mask = mask & ~interior

    res = np.full_like(arr, bg)
    res[outline_mask] = out_col
    return ArcGrid(pixels=res, background=bg)


@_register("fill_enclosed_holes")
def fill_enclosed_holes(grid: ArcGrid, fill_color: int = 1) -> ArcGrid:
    """
    Fills enclosed background regions (holes surrounded by non-background pixels)
    with *fill_color*.
    """
    arr = grid.pixels.copy()
    bg = grid.background
    h, w = arr.shape

    # Flood fill exterior background from grid borders
    from collections import deque
    exterior_bg = np.zeros((h, w), dtype=bool)
    queue = deque()

    for r in range(h):
        for c in (0, w - 1):
            if arr[r, c] == bg and not exterior_bg[r, c]:
                exterior_bg[r, c] = True
                queue.append((r, c))

    for c in range(w):
        for r in (0, h - 1):
            if arr[r, c] == bg and not exterior_bg[r, c]:
                exterior_bg[r, c] = True
                queue.append((r, c))

    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    while queue:
        r, c = queue.popleft()
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]
            if 0 <= nr < h and 0 <= nc < w:
                if arr[nr, nc] == bg and not exterior_bg[nr, nc]:
                    exterior_bg[nr, nc] = True
                    queue.append((nr, nc))

    # Holes are background pixels that are not exterior
    holes_mask = (arr == bg) & ~exterior_bg
    res = arr.copy()
    res[holes_mask] = fill_color
    return ArcGrid(pixels=res, background=bg)


@_register("pattern_repeat")
def pattern_repeat(grid: ArcGrid, tile_h: int = 2, tile_w: int = 2) -> ArcGrid:
    """
    Repeats the grid content as a periodic tile to fill a larger canvas.
    """
    arr = grid.pixels.copy()
    tiled = np.tile(arr, (tile_h, tile_w))
    return ArcGrid(pixels=tiled, background=grid.background)


@_register("step_ca_majority")
def step_ca_majority(grid: ArcGrid) -> ArcGrid:
    """
    Executes a 1-step majority-vote cellular automaton over non-background neighbors.
    Useful for smoothing or noise-reduction tasks.
    """
    arr = grid.pixels.copy()
    bg = grid.background
    h, w = arr.shape
    res = arr.copy()

    for r in range(h):
        for c in range(w):
            neighbors = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbors.append(arr[nr, nc])
            if neighbors:
                counts = {}
                for val in neighbors:
                    counts[val] = counts.get(val, 0) + 1
                most_common = max(counts.items(), key=lambda x: x[1])[0]
                if counts[most_common] >= 5:
                    res[r, c] = most_common

    return ArcGrid(pixels=res, background=bg)
