"""
Higher-Order AST Program Primitives — Object-level mapping, conditional transformations,
and line reductions.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import numpy as np

from src.core.grid.grid import ArcGrid
from src.core.objects.detector import ObjectDetector
from src.dsl.primitives.transforms import _register, PRIMITIVE_REGISTRY


@_register("map_objects")
def map_objects(grid: ArcGrid, primitive_name: str = "rotate_90") -> ArcGrid:
    """
    Detects objects and applies specified primitive function independently to each object.
    """
    if primitive_name not in PRIMITIVE_REGISTRY:
        return grid

    primitive_fn = PRIMITIVE_REGISTRY[primitive_name]
    detector = ObjectDetector()
    objects = detector.detect(grid)

    if not objects:
        return grid

    res_arr = np.full_like(grid.pixels, grid.background)

    for obj in objects:
        bbox = obj.bounding_box
        sub_arr = grid.pixels[bbox.row_min:bbox.row_max + 1, bbox.col_min:bbox.col_max + 1].copy()
        mask = np.zeros_like(sub_arr, dtype=bool)
        for r, c in obj.pixels:
            mask[r - bbox.row_min, c - bbox.col_min] = True

        # Mask background
        sub_arr[~mask] = grid.background
        sub_grid = ArcGrid(pixels=sub_arr, background=grid.background)

        try:
            trans_grid = primitive_fn(sub_grid)
            th, tw = trans_grid.pixels.shape
            r_end = min(bbox.row_min + th, grid.height)
            c_end = min(bbox.col_min + tw, grid.width)
            sub_trans = trans_grid.pixels[:r_end - bbox.row_min, :c_end - bbox.col_min]

            t_mask = (sub_trans != grid.background)
            res_arr[bbox.row_min:r_end, bbox.col_min:c_end][t_mask] = sub_trans[t_mask]
        except Exception:
            for r, c in obj.pixels:
                res_arr[r, c] = obj.color

    return ArcGrid(pixels=res_arr, background=grid.background)


@_register("conditional_apply")
def conditional_apply(
    grid: ArcGrid,
    condition: str = "is_symmetric",
    true_primitive: str = "mirror_horizontal",
    false_primitive: str = "rotate_90",
) -> ArcGrid:
    """
    Applies true_primitive if grid satisfies condition (e.g. is_symmetric, has_holes),
    else applies false_primitive.
    """
    arr = grid.pixels.copy()
    bg = grid.background

    is_true = False
    if condition == "is_symmetric":
        is_true = (arr == np.fliplr(arr)).all() or (arr == np.flipud(arr)).all()
    elif condition == "has_holes":
        from src.dsl.primitives.advanced_primitives import fill_enclosed_holes
        filled = fill_enclosed_holes(grid)
        is_true = (filled.pixels != arr).any()

    target_primitive = true_primitive if is_true else false_primitive
    if target_primitive in PRIMITIVE_REGISTRY:
        return PRIMITIVE_REGISTRY[target_primitive](grid)

    return grid


@_register("reduce_lines")
def reduce_lines(grid: ArcGrid, axis: int = 0, mode: str = "or") -> ArcGrid:
    """
    Reduces grid along axis (0 for rows, 1 for columns) using logical/bitwise combination.
    """
    arr = grid.pixels.copy()
    bg = grid.background

    if axis == 0:
        # Collapse rows -> 1 x W
        res = np.zeros((1, arr.shape[1]), dtype=int)
        for c in range(arr.shape[1]):
            col = arr[:, c]
            non_bg = col[col != bg]
            res[0, c] = non_bg[0] if len(non_bg) > 0 else bg
    else:
        # Collapse columns -> H x 1
        res = np.zeros((arr.shape[0], 1), dtype=int)
        for r in range(arr.shape[0]):
            row = arr[r, :]
            non_bg = row[row != bg]
            res[r, 0] = non_bg[0] if len(non_bg) > 0 else bg

    return ArcGrid(pixels=res, background=bg)
