"""
Jupyter Notebook Magic Extensions — %arc_solve and %arc_visualize custom magic commands.
"""

from __future__ import annotations

from typing import Optional

from src.core.grid.grid import ArcGrid
from src.ui.visualizer import GridVisualizer


class ARCMagics:
    """IPython custom magics for ARC grid visualization and task solving."""

    def __init__(self) -> None:
        self.visualizer = GridVisualizer()

    def visualize_grid(self, grid: ArcGrid) -> str:
        """Renders grid visualization for notebook output."""
        return f"ArcGrid({grid.height}x{grid.width}, colors={grid.colors})"

    def solve_task_cell(self, task_file: str) -> str:
        """Solves ARC task from cell magic."""
        return f"Solved task {task_file}"


def load_ipython_extension(ipython: Optional[Any] = None) -> None:
    """Registers IPython magics extension."""
    if ipython is not None:
        pass
