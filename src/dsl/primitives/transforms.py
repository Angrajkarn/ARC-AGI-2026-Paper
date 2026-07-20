"""
DSL Primitives — ~35 atomic grid transformation functions.

Each primitive is a pure function:
    (ArcGrid, **kwargs) → ArcGrid

All primitives are registered in PRIMITIVE_REGISTRY for the DSL executor.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid
from src.core.objects.arc_object import ArcObject
from src.core.objects.detector import ObjectDetector

# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

PRIMITIVE_REGISTRY: Dict[str, Callable[..., ArcGrid]] = {}


def _register(name: str):
    """Decorator to register a primitive function by name."""
    def decorator(fn: Callable) -> Callable:
        PRIMITIVE_REGISTRY[name] = fn
        return fn
    return decorator


# ------------------------------------------------------------------
# Rotation & Reflection
# ------------------------------------------------------------------

@_register("rotate_90")
def rotate_90(grid: ArcGrid, times: int = 1) -> ArcGrid:
    """Rotate grid 90° clockwise *times* times."""
    arr = grid.pixels.copy()
    for _ in range(times % 4):
        arr = np.rot90(arr, k=-1)  # k=-1 = 90° clockwise
    return ArcGrid(pixels=arr, background=grid.background)


@_register("rotate_180")
def rotate_180(grid: ArcGrid) -> ArcGrid:
    return rotate_90(grid, times=2)


@_register("rotate_270")
def rotate_270(grid: ArcGrid) -> ArcGrid:
    return rotate_90(grid, times=3)


@_register("mirror_horizontal")
def mirror_horizontal(grid: ArcGrid) -> ArcGrid:
    """Flip left-right."""
    return ArcGrid(pixels=np.fliplr(grid.pixels.copy()), background=grid.background)


@_register("mirror_vertical")
def mirror_vertical(grid: ArcGrid) -> ArcGrid:
    """Flip top-bottom."""
    return ArcGrid(pixels=np.flipud(grid.pixels.copy()), background=grid.background)


@_register("mirror_diagonal")
def mirror_diagonal(grid: ArcGrid) -> ArcGrid:
    """Transpose (mirror along main diagonal)."""
    return ArcGrid(pixels=grid.pixels.T.copy(), background=grid.background)


@_register("mirror_antidiagonal")
def mirror_antidiagonal(grid: ArcGrid) -> ArcGrid:
    """Mirror along the anti-diagonal."""
    return ArcGrid(pixels=np.rot90(np.fliplr(grid.pixels.copy()), k=1),
                   background=grid.background)


# ------------------------------------------------------------------
# Crop & Resize
# ------------------------------------------------------------------

@_register("crop")
def crop(grid: ArcGrid, row_min: int, row_max: int, col_min: int, col_max: int) -> ArcGrid:
    """Crop to a rectangular region (inclusive bounds)."""
    arr = grid.pixels[row_min:row_max + 1, col_min:col_max + 1].copy()
    return ArcGrid(pixels=arr, background=grid.background)


@_register("crop_to_content")
def crop_to_content(grid: ArcGrid) -> ArcGrid:
    """Crop to the tightest bounding box around non-background pixels."""
    rows, cols = np.where(grid.pixels != grid.background)
    if len(rows) == 0:
        return grid.copy()
    return crop(grid, int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max()))


@_register("scale")
def scale(grid: ArcGrid, factor: int) -> ArcGrid:
    """Upscale grid by integer factor (nearest-neighbour)."""
    arr = np.repeat(np.repeat(grid.pixels, factor, axis=0), factor, axis=1)
    return ArcGrid(pixels=arr, background=grid.background)


@_register("resize")
def resize(grid: ArcGrid, new_height: int, new_width: int) -> ArcGrid:
    """Resize grid to (new_height, new_width) via nearest-neighbour."""
    from scipy.ndimage import zoom
    arr = grid.pixels.astype(float)
    zy = new_height / grid.height
    zx = new_width / grid.width
    resized = zoom(arr, (zy, zx), order=0).astype(np.int8)
    resized = resized[:new_height, :new_width]  # guard against off-by-one
    return ArcGrid(pixels=resized, background=grid.background)


# ------------------------------------------------------------------
# Translation
# ------------------------------------------------------------------

@_register("translate")
def translate(grid: ArcGrid, dr: int, dc: int, fill: Optional[int] = None) -> ArcGrid:
    """Shift all non-background pixels by (dr, dc), filling vacated cells."""
    if fill is None:
        fill = grid.background
    H, W = grid.height, grid.width
    arr = np.full((H, W), fill, dtype=np.int8)
    for r in range(H):
        for c in range(W):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                arr[nr, nc] = grid.get(r, c)
    return ArcGrid(pixels=arr, background=grid.background)


# ------------------------------------------------------------------
# Repetition
# ------------------------------------------------------------------

@_register("tile")
def tile(grid: ArcGrid, rows: int, cols: int) -> ArcGrid:
    """Tile the grid into an (rows x cols) mosaic."""
    arr = np.tile(grid.pixels, (rows, cols))
    return ArcGrid(pixels=arr, background=grid.background)


@_register("repeat_pattern")
def repeat_pattern(grid: ArcGrid, direction: str = "right", times: int = 1) -> ArcGrid:
    """Repeat the grid content along a direction."""
    if direction == "right":
        arr = np.concatenate([grid.pixels] * (times + 1), axis=1)
    elif direction == "down":
        arr = np.concatenate([grid.pixels] * (times + 1), axis=0)
    else:
        raise ValueError(f"Unknown direction: {direction!r}")
    return ArcGrid(pixels=arr, background=grid.background)


# ------------------------------------------------------------------
# Colour Operations
# ------------------------------------------------------------------

@_register("replace_color")
def replace_color(grid: ArcGrid, source_color: int, target_color: int) -> ArcGrid:
    """Replace all pixels of source_color with target_color."""
    arr = grid.pixels.copy()
    arr[arr == source_color] = target_color
    bg = target_color if grid.background == source_color else grid.background
    return ArcGrid(pixels=arr, background=bg)


@_register("swap_colors")
def swap_colors(grid: ArcGrid, color_a: int, color_b: int) -> ArcGrid:
    """Swap two colours throughout the grid."""
    arr = grid.pixels.copy()
    mask_a = arr == color_a
    mask_b = arr == color_b
    arr[mask_a] = color_b
    arr[mask_b] = color_a
    return ArcGrid(pixels=arr, background=grid.background)


@_register("normalize_colors")
def normalize_colors(grid: ArcGrid) -> ArcGrid:
    """Remap colours to canonical form (most common → 0, next → 1, etc.)."""
    values, counts = np.unique(grid.pixels, return_counts=True)
    sorted_vals = values[np.argsort(-counts)]
    mapping = {int(old): new for new, old in enumerate(sorted_vals)}
    arr = np.vectorize(lambda x: mapping[x])(grid.pixels).astype(np.int8)
    return ArcGrid(pixels=arr, background=mapping[grid.background])


# ------------------------------------------------------------------
# Fill & Flood Operations
# ------------------------------------------------------------------

@_register("flood_fill")
def flood_fill(grid: ArcGrid, row: int, col: int, new_color: int) -> ArcGrid:
    """Flood fill from (row, col) with new_color (4-connected)."""
    from collections import deque
    arr = grid.pixels.copy()
    target = int(arr[row, col])
    if target == new_color:
        return ArcGrid(pixels=arr, background=grid.background)
    queue: deque = deque([(row, col)])
    while queue:
        r, c = queue.popleft()
        if not (0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]):
            continue
        if arr[r, c] != target:
            continue
        arr[r, c] = new_color
        queue.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
    return ArcGrid(pixels=arr, background=grid.background)


@_register("fill_holes")
def fill_holes(grid: ArcGrid, fill_color: Optional[int] = None) -> ArcGrid:
    """Fill enclosed background holes within objects."""
    from src.core.objects.detector import ObjectDetector
    detector = ObjectDetector()
    objects = detector.detect(grid)
    arr = grid.pixels.copy()
    fc = fill_color if fill_color is not None else grid.background
    for obj in objects:
        for hole in obj.holes:
            for r, c in hole:
                arr[r, c] = fc if fill_color is not None else obj.color
    return ArcGrid(pixels=arr, background=grid.background)


# ------------------------------------------------------------------
# Border & Outline
# ------------------------------------------------------------------

@_register("add_border")
def add_border(grid: ArcGrid, color: int, thickness: int = 1) -> ArcGrid:
    """Add a uniform border of given colour and thickness around the grid."""
    H, W = grid.height, grid.width
    t = thickness
    arr = np.full((H + 2 * t, W + 2 * t), color, dtype=np.int8)
    arr[t:t + H, t:t + W] = grid.pixels
    return ArcGrid(pixels=arr, background=grid.background)


@_register("outline")
def outline(grid: ArcGrid, color: int) -> ArcGrid:
    """Draw an outline (1-px border) around all non-background objects."""
    arr = grid.pixels.copy()
    H, W = arr.shape
    new_arr = arr.copy()
    for r in range(H):
        for c in range(W):
            if arr[r, c] == grid.background:
                # Check if any neighbour is non-background
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and arr[nr, nc] != grid.background:
                        new_arr[r, c] = color
                        break
    return ArcGrid(pixels=new_arr, background=grid.background)


@_register("hollow")
def hollow(grid: ArcGrid) -> ArcGrid:
    """Keep only the border pixels of each object (make them hollow)."""
    arr = grid.pixels.copy()
    H, W = arr.shape
    new_arr = np.full_like(arr, grid.background)
    for r in range(H):
        for c in range(W):
            if arr[r, c] == grid.background:
                continue
            # Keep if it has at least one background neighbour
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W) or arr[nr, nc] == grid.background:
                    new_arr[r, c] = arr[r, c]
                    break
    return ArcGrid(pixels=new_arr, background=grid.background)


# ------------------------------------------------------------------
# Object-level Operations
# ------------------------------------------------------------------

@_register("delete_object")
def delete_object(grid: ArcGrid, color: int) -> ArcGrid:
    """Delete all pixels of a given colour (replace with background)."""
    return replace_color(grid, color, grid.background)


@_register("keep_object")
def keep_object(grid: ArcGrid, color: int) -> ArcGrid:
    """Keep only pixels of the given colour; background everything else."""
    arr = np.where(grid.pixels == color, grid.pixels, grid.background).astype(np.int8)
    return ArcGrid(pixels=arr, background=grid.background)


@_register("duplicate_object")
def duplicate_object(grid: ArcGrid, color: int, dr: int, dc: int) -> ArcGrid:
    """Duplicate the object of *color* shifted by (dr, dc)."""
    arr = grid.pixels.copy()
    positions = list(zip(*np.where(arr == color)))
    for r, c in positions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < arr.shape[0] and 0 <= nc < arr.shape[1]:
            arr[nr, nc] = color
    return ArcGrid(pixels=arr, background=grid.background)


@_register("move_object")
def move_object(grid: ArcGrid, color: int, dr: int, dc: int) -> ArcGrid:
    """Move the object of *color* by (dr, dc), erasing original position."""
    arr = grid.pixels.copy()
    positions = list(zip(*np.where(arr == color)))
    for r, c in positions:
        arr[r, c] = grid.background
    for r, c in positions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < arr.shape[0] and 0 <= nc < arr.shape[1]:
            arr[nr, nc] = color
    return ArcGrid(pixels=arr, background=grid.background)


@_register("extract_object")
def extract_object(grid: ArcGrid, color: int) -> ArcGrid:
    """Return a grid cropped to the bounding box of the *color* object."""
    positions = grid.pixels_of_color(color)
    if not positions:
        return grid.copy()
    rows = [r for r, _ in positions]
    cols = [c for _, c in positions]
    return crop(grid, min(rows), max(rows), min(cols), max(cols))


@_register("expand_object")
def expand_object(grid: ArcGrid, color: int, radius: int = 1) -> ArcGrid:
    """Dilate the object of *color* by *radius* pixels (4-connected)."""
    from collections import deque
    arr = grid.pixels.copy()
    seeds = list(zip(*np.where(arr == color)))
    front = deque(seeds)
    visited = set(seeds)
    for _ in range(radius):
        next_front: deque = deque()
        while front:
            r, c = front.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < arr.shape[0]
                    and 0 <= nc < arr.shape[1]
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    arr[nr, nc] = color
                    next_front.append((nr, nc))
        front = next_front
    return ArcGrid(pixels=arr, background=grid.background)


@_register("shrink_object")
def shrink_object(grid: ArcGrid, color: int, radius: int = 1) -> ArcGrid:
    """Erode the object of *color* by *radius* pixels (4-connected)."""
    arr = grid.pixels.copy()
    H, W = arr.shape
    for _ in range(radius):
        to_remove = []
        for r in range(H):
            for c in range(W):
                if arr[r, c] != color:
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W) or arr[nr, nc] != color:
                        to_remove.append((r, c))
                        break
        for r, c in to_remove:
            arr[r, c] = grid.background
    return ArcGrid(pixels=arr, background=grid.background)


# ------------------------------------------------------------------
# Grid Composition
# ------------------------------------------------------------------

@_register("union_grids")
def union_grids(grid_a: ArcGrid, grid_b: ArcGrid, priority: str = "a") -> ArcGrid:
    """Overlay two grids of the same size; non-background pixels take priority."""
    assert grid_a.size == grid_b.size, "Grids must have identical dimensions"
    arr = grid_a.pixels.copy()
    bg = grid_a.background
    mask_b_non_bg = grid_b.pixels != grid_b.background
    if priority == "b":
        arr[mask_b_non_bg] = grid_b.pixels[mask_b_non_bg]
    else:
        # a takes priority: only fill background cells with b
        arr[arr == bg] = grid_b.pixels[arr == bg]
    return ArcGrid(pixels=arr, background=bg)


@_register("intersect_grids")
def intersect_grids(grid_a: ArcGrid, grid_b: ArcGrid) -> ArcGrid:
    """Keep only pixels present (non-background) in BOTH grids."""
    assert grid_a.size == grid_b.size
    arr = np.where(
        (grid_a.pixels != grid_a.background) & (grid_b.pixels != grid_b.background),
        grid_a.pixels,
        grid_a.background,
    ).astype(np.int8)
    return ArcGrid(pixels=arr, background=grid_a.background)


@_register("diff_grids")
def diff_grids(grid_a: ArcGrid, grid_b: ArcGrid) -> ArcGrid:
    """Pixels in A but not in B (set difference)."""
    assert grid_a.size == grid_b.size
    arr = np.where(
        (grid_a.pixels != grid_a.background) & (grid_b.pixels == grid_b.background),
        grid_a.pixels,
        grid_a.background,
    ).astype(np.int8)
    return ArcGrid(pixels=arr, background=grid_a.background)


@_register("mask")
def mask(grid: ArcGrid, mask_grid: ArcGrid, mask_color: int) -> ArcGrid:
    """Zero out (background) pixels where mask_grid == mask_color."""
    assert grid.size == mask_grid.size
    arr = grid.pixels.copy()
    arr[mask_grid.pixels == mask_color] = grid.background
    return ArcGrid(pixels=arr, background=grid.background)


# ------------------------------------------------------------------
# Structural / Spatial Operations
# ------------------------------------------------------------------

@_register("gravity")
def gravity(grid: ArcGrid, direction: str = "down") -> ArcGrid:
    """Apply gravity: non-background pixels fall in *direction*."""
    arr = grid.pixels.copy()
    H, W = arr.shape
    bg = grid.background

    if direction == "down":
        for c in range(W):
            col = [arr[r, c] for r in range(H) if arr[r, c] != bg]
            col = [bg] * (H - len(col)) + col
            for r, v in enumerate(col):
                arr[r, c] = v
    elif direction == "up":
        for c in range(W):
            col = [arr[r, c] for r in range(H) if arr[r, c] != bg]
            col = col + [bg] * (H - len(col))
            for r, v in enumerate(col):
                arr[r, c] = v
    elif direction == "right":
        for r in range(H):
            row = [arr[r, c] for c in range(W) if arr[r, c] != bg]
            row = [bg] * (W - len(row)) + row
            for c, v in enumerate(row):
                arr[r, c] = v
    elif direction == "left":
        for r in range(H):
            row = [arr[r, c] for c in range(W) if arr[r, c] != bg]
            row = row + [bg] * (W - len(row))
            for c, v in enumerate(row):
                arr[r, c] = v
    else:
        raise ValueError(f"Unknown direction: {direction!r}")

    return ArcGrid(pixels=arr, background=bg)


@_register("sort_objects")
def sort_objects(grid: ArcGrid, key: str = "area", reverse: bool = True) -> ArcGrid:
    """Sort detected objects by *key* and re-draw them top-to-bottom."""
    from src.core.objects.detector import ObjectDetector
    detector = ObjectDetector()
    objects = detector.detect(grid)
    key_fn = {
        "area": lambda o: o.area,
        "color": lambda o: o.color,
        "x": lambda o: o.bounding_box.col_min,
        "y": lambda o: o.bounding_box.row_min,
    }.get(key, lambda o: o.area)
    objects.sort(key=key_fn, reverse=reverse)

    arr = np.full_like(grid.pixels, grid.background)
    cur_row = 0
    for obj in objects:
        snippet = obj.to_grid_snippet()
        h, w = snippet.shape
        if cur_row + h > arr.shape[0]:
            break
        arr[cur_row:cur_row + h, :w] = snippet
        cur_row += h
    return ArcGrid(pixels=arr, background=grid.background)


@_register("align_objects")
def align_objects(grid: ArcGrid, alignment: str = "left") -> ArcGrid:
    """Align all detected objects to a common edge."""
    from src.core.objects.detector import ObjectDetector
    detector = ObjectDetector()
    objects = detector.detect(grid)
    arr = np.full_like(grid.pixels, grid.background)
    H, W = arr.shape
    for obj in objects:
        snippet = obj.to_grid_snippet()
        h, w = snippet.shape
        r0 = obj.bounding_box.row_min
        if alignment == "left":
            arr[r0:r0 + h, :w] = snippet
        elif alignment == "right":
            arr[r0:r0 + h, W - w:W] = snippet
        elif alignment == "top":
            arr[:h, obj.bounding_box.col_min:obj.bounding_box.col_min + w] = snippet
        elif alignment == "bottom":
            arr[H - h:H, obj.bounding_box.col_min:obj.bounding_box.col_min + w] = snippet
    return ArcGrid(pixels=arr, background=grid.background)


@_register("reflect")
def reflect(grid: ArcGrid, axis: str = "vertical") -> ArcGrid:
    """Reflect and concatenate the grid along an axis to create symmetry."""
    if axis == "vertical":
        mirrored = mirror_horizontal(grid)
        arr = np.concatenate([grid.pixels, mirrored.pixels], axis=1)
    elif axis == "horizontal":
        mirrored = mirror_vertical(grid)
        arr = np.concatenate([grid.pixels, mirrored.pixels], axis=0)
    else:
        raise ValueError(f"Unknown axis: {axis!r}")
    return ArcGrid(pixels=arr, background=grid.background)


@_register("connect_objects")
def connect_objects(grid: ArcGrid, color_a: int, color_b: int, line_color: int) -> ArcGrid:
    """Draw a straight line between the centroids of two colour objects."""
    pos_a = grid.pixels_of_color(color_a)
    pos_b = grid.pixels_of_color(color_b)
    if not pos_a or not pos_b:
        return grid.copy()

    ar = sum(r for r, _ in pos_a) / len(pos_a)
    ac = sum(c for _, c in pos_a) / len(pos_a)
    br = sum(r for r, _ in pos_b) / len(pos_b)
    bc = sum(c for _, c in pos_b) / len(pos_b)

    arr = grid.pixels.copy()
    steps = max(abs(int(br) - int(ar)), abs(int(bc) - int(ac)), 1)
    for i in range(steps + 1):
        t = i / steps
        r = round(ar + t * (br - ar))
        c = round(ac + t * (bc - ac))
        if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
            arr[r, c] = line_color
    return ArcGrid(pixels=arr, background=grid.background)


@_register("split")
def split(grid: ArcGrid, axis: str = "horizontal", parts: int = 2) -> List[ArcGrid]:
    """Split grid into equal parts along axis. Returns list of ArcGrids."""
    arr = grid.pixels
    if axis == "horizontal":
        splits = np.array_split(arr, parts, axis=0)
    elif axis == "vertical":
        splits = np.array_split(arr, parts, axis=1)
    else:
        raise ValueError(f"Unknown axis: {axis!r}")
    return [ArcGrid(pixels=s.copy(), background=grid.background) for s in splits]


@_register("merge_grids")
def merge_grids(grids: List[ArcGrid], axis: str = "horizontal") -> ArcGrid:
    """Concatenate a list of grids along an axis."""
    arrs = [g.pixels for g in grids]
    if axis == "horizontal":
        arr = np.concatenate(arrs, axis=0)
    elif axis == "vertical":
        arr = np.concatenate(arrs, axis=1)
    else:
        raise ValueError(f"Unknown axis: {axis!r}")
    return ArcGrid(pixels=arr, background=grids[0].background)


@_register("compose")
def compose(grid: ArcGrid, operations: List[Dict]) -> ArcGrid:
    """Apply a sequence of operations to a grid (sequential composition).

    Args:
        grid:       Input grid.
        operations: List of {"name": str, "args": dict} dicts.

    Returns:
        Resulting grid after all operations.
    """
    result = grid
    for op in operations:
        name = op["name"]
        args = op.get("args", {})
        if name not in PRIMITIVE_REGISTRY:
            raise ValueError(f"Unknown primitive: {name!r}")
        result = PRIMITIVE_REGISTRY[name](result, **args)
    return result


@_register("invert_colors")
def invert_colors(grid: ArcGrid) -> ArcGrid:
    """Invert colors: 0 -> 9, 1 -> 8, ..., 9 -> 0."""
    out = grid.copy()
    out.pixels = (9 - grid.pixels).astype(np.int8)
    return out


@_register("recolor_non_background")
def recolor_non_background(grid: ArcGrid, new_color: int = 1) -> ArcGrid:
    """Replace all non-background pixels with new_color."""
    out = grid.copy()
    mask = grid.pixels != grid.background
    out.pixels[mask] = new_color
    return out


@_register("isolate_largest")
def isolate_largest(grid: ArcGrid) -> ArcGrid:
    """Keep only the largest object in the grid, backgrounding all else."""
    detector = ObjectDetector()
    objects = detector.detect(grid)
    if not objects:
        return grid.copy()
    largest = max(objects, key=lambda o: o.area)
    out = ArcGrid.empty(grid.height, grid.width, fill=grid.background)
    for r, c in largest.pixels:
        out.set(r, c, largest.color)
    return out


@_register("isolate_smallest")
def isolate_smallest(grid: ArcGrid) -> ArcGrid:
    """Keep only the smallest object in the grid, backgrounding all else."""
    detector = ObjectDetector()
    objects = detector.detect(grid)
    if not objects:
        return grid.copy()
    smallest = min(objects, key=lambda o: o.area)
    out = ArcGrid.empty(grid.height, grid.width, fill=grid.background)
    for r, c in smallest.pixels:
        out.set(r, c, smallest.color)
    return out


@_register("center_content")
def center_content(grid: ArcGrid) -> ArcGrid:
    """Center non-background content in the grid canvas."""
    non_bg = grid.pixels != grid.background
    if not non_bg.any():
        return grid.copy()
    rows, cols = np.where(non_bg)
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    ch, cw = rmax - rmin + 1, cmax - cmin + 1

    content = grid.pixels[rmin:rmax + 1, cmin:cmax + 1]
    out = ArcGrid.empty(grid.height, grid.width, fill=grid.background)

    start_r = (grid.height - ch) // 2
    start_c = (grid.width - cw) // 2
    out.pixels[start_r:start_r + ch, start_c:start_c + cw] = content
    return out


__all__ = ["PRIMITIVE_REGISTRY"] + list(PRIMITIVE_REGISTRY.keys())
