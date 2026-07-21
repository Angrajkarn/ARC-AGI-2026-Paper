"""
Hierarchical Cellular Automata & Physics Primitives — Sand flow, water fill, heat diffusion, Game of Life.
"""

from __future__ import annotations

import numpy as np

from src.core.grid.grid import ArcGrid
from src.dsl.primitives.transforms import PRIMITIVE_REGISTRY


def ca_sand_fall(grid: ArcGrid, sand_color: int = 1, obstacle_color: int = 2) -> ArcGrid:
    """Simulates 1-step sand grain falling down under gravity."""
    res = grid.copy()
    h, w = grid.height, grid.width

    for r in range(h - 2, -1, -1):
        for c in range(w):
            if res.get(r, c) == sand_color and res.get(r + 1, c) == grid.background:
                res.set(r, c, grid.background)
                res.set(r + 1, c, sand_color)

    return res


def ca_water_fill(grid: ArcGrid, water_color: int = 8, fill_color: int = 1) -> ArcGrid:
    """Fills adjacent background cells surrounding water_color pixels."""
    res = grid.copy()
    h, w = grid.height, grid.width

    for r in range(h):
        for c in range(w):
            if grid.get(r, c) == water_color:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid.get(nr, nc) == grid.background:
                        res.set(nr, nc, fill_color)

    return res


def ca_game_of_life(grid: ArcGrid, live_color: int = 1) -> ArcGrid:
    """Simulates 1 step of Conway's Game of Life for live_color pixels."""
    res = grid.copy()
    h, w = grid.height, grid.width

    for r in range(h):
        for c in range(w):
            neighbors = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid.get(nr, nc) == live_color:
                        neighbors += 1

            is_alive = (grid.get(r, c) == live_color)
            if is_alive and (neighbors < 2 or neighbors > 3):
                res.set(r, c, grid.background)
            elif not is_alive and neighbors == 3:
                res.set(r, c, live_color)

    return res


# Register in global PRIMITIVE_REGISTRY
PRIMITIVE_REGISTRY["ca_sand_fall"] = ca_sand_fall
PRIMITIVE_REGISTRY["ca_water_fill"] = ca_water_fill
PRIMITIVE_REGISTRY["ca_game_of_life"] = ca_game_of_life
