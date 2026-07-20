"""
Higher-Order DSL Combinators — composite operations over objects, layers, and regions.

Combinators allow high-level declarative logic:
  - Object filtering by size, color, or shape
  - Object mapping & recoloring
  - Multi-layer overlaying and underlaying
  - Line drawing and path connecting
"""

from __future__ import annotations

from typing import Callable, List, Optional, Set, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid
from src.core.objects.arc_object import ArcObject
from src.core.objects.detector import ObjectDetector
from src.dsl.primitives.transforms import PRIMITIVE_REGISTRY


# ---------------------------------------------------------------------------
# Higher-Order Primitives
# ---------------------------------------------------------------------------

def filter_objects_by_color(grid: ArcGrid, color: int) -> ArcGrid:
    """Keep only objects of the specified colour, setting others to background."""
    detector = ObjectDetector()
    objects = detector.detect(grid)
    out = ArcGrid.empty(grid.height, grid.width, fill=grid.background)
    for obj in objects:
        if obj.color == color:
            for r, c in obj.pixels:
                out.set(r, c, obj.color)
    return out


def filter_objects_by_size(grid: ArcGrid, mode: str = "largest") -> ArcGrid:
    """Keep only the 'largest' or 'smallest' object(s)."""
    detector = ObjectDetector()
    objects = detector.detect(grid)
    if not objects:
        return grid.copy()

    areas = [o.area for o in objects]
    target_area = max(areas) if mode == "largest" else min(areas)

    out = ArcGrid.empty(grid.height, grid.width, fill=grid.background)
    for obj in objects:
        if obj.area == target_area:
            for r, c in obj.pixels:
                out.set(r, c, obj.color)
    return out


def filter_objects_by_shape(grid: ArcGrid, shape_type: str = "rectangle") -> ArcGrid:
    """Keep only objects matching shape_type ('rectangle', 'line')."""
    detector = ObjectDetector()
    objects = detector.detect(grid)
    out = ArcGrid.empty(grid.height, grid.width, fill=grid.background)
    for obj in objects:
        if shape_type in obj.tags:
            for r, c in obj.pixels:
                out.set(r, c, obj.color)
    return out


def overlay(grid_a: ArcGrid, grid_b: ArcGrid, transparent_color: int = 0) -> ArcGrid:
    """Overlay grid_b on top of grid_a where grid_b != transparent_color."""
    h = max(grid_a.height, grid_b.height)
    w = max(grid_a.width, grid_b.width)
    out = ArcGrid.empty(h, w, fill=grid_a.background)

    # Copy grid_a
    out.pixels[:grid_a.height, :grid_a.width] = grid_a.pixels.copy()

    # Overlay grid_b non-transparent pixels
    mask = grid_b.pixels != transparent_color
    out.pixels[:grid_b.height, :grid_b.width][mask] = grid_b.pixels[mask]
    return out


def connect_points_of_color(grid: ArcGrid, color: int, line_color: Optional[int] = None) -> ArcGrid:
    """Draw straight horizontal/vertical connecting lines between pixels of *color*."""
    if line_color is None:
        line_color = color

    positions = grid.pixels_of_color(color)
    if len(positions) < 2:
        return grid.copy()

    out = grid.copy()
    # Check all pairs for horizontal or vertical alignment
    for i in range(len(positions)):
        r1, c1 = positions[i]
        for j in range(i + 1, len(positions)):
            r2, c2 = positions[j]
            if r1 == r2:
                # Same row -> draw horizontal segment
                cmin, cmax = min(c1, c2), max(c1, c2)
                out.pixels[r1, cmin:cmax + 1] = line_color
            elif c1 == c2:
                # Same col -> draw vertical segment
                rmin, rmax = min(r1, r2), max(r1, r2)
                out.pixels[rmin:rmax + 1, c1] = line_color

    return out


def draw_line(
    grid: ArcGrid, r1: int, c1: int, r2: int, c2: int, color: int
) -> ArcGrid:
    """Draw a line segment from (r1, c1) to (r2, c2) using Bresenham's algorithm."""
    out = grid.copy()
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    sr = 1 if r1 < r2 else -1
    sc = 1 if c1 < c2 else -1
    err = dr - dc

    r, c = r1, c1
    while True:
        if out.in_bounds(r, c):
            out.set(r, c, color)
        if r == r2 and c == c2:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

    return out


def complete_symmetry_h(grid: ArcGrid) -> ArcGrid:
    """Make the grid horizontally symmetric by mirroring non-background content."""
    out = grid.copy()
    w = grid.width
    for r in range(grid.height):
        for c in range(w // 2):
            left_val = grid.get(r, c)
            right_val = grid.get(r, w - 1 - c)
            if left_val != grid.background and right_val == grid.background:
                out.set(r, w - 1 - c, left_val)
            elif right_val != grid.background and left_val == grid.background:
                out.set(r, c, right_val)
    return out


def complete_symmetry_v(grid: ArcGrid) -> ArcGrid:
    """Make the grid vertically symmetric by mirroring non-background content."""
    out = grid.copy()
    h = grid.height
    for r in range(h // 2):
        for c in range(grid.width):
            top_val = grid.get(r, c)
            bot_val = grid.get(h - 1 - r, c)
            if top_val != grid.background and bot_val == grid.background:
                out.set(h - 1 - r, c, top_val)
            elif bot_val != grid.background and top_val == grid.background:
                out.set(r, c, bot_val)
    return out


# Register all combinators into PRIMITIVE_REGISTRY
PRIMITIVE_REGISTRY.update({
    "filter_objects_by_color": filter_objects_by_color,
    "filter_objects_by_size": filter_objects_by_size,
    "filter_objects_by_shape": filter_objects_by_shape,
    "overlay": overlay,
    "connect_points_of_color": connect_points_of_color,
    "draw_line": draw_line,
    "complete_symmetry_h": complete_symmetry_h,
    "complete_symmetry_v": complete_symmetry_v,
})

__all__ = [
    "filter_objects_by_color",
    "filter_objects_by_size",
    "filter_objects_by_shape",
    "overlay",
    "connect_points_of_color",
    "draw_line",
    "complete_symmetry_h",
    "complete_symmetry_v",
]
