"""
Grid Visualizer — terminal and matplotlib visualizations of ARC grids.

Features:
  - Terminal colour-coded output
  - Side-by-side input/output/prediction display
  - Matplotlib figure generation (for notebooks)
"""

from __future__ import annotations

from typing import List, Optional

from src.core.grid.grid import ArcGrid, COLOR_NAMES


# ANSI colour codes for terminal display (approximate mapping)
_ANSI_COLORS = {
    0: "\033[40m",   # black
    1: "\033[44m",   # blue
    2: "\033[41m",   # red
    3: "\033[42m",   # green
    4: "\033[43m",   # yellow
    5: "\033[47m",   # light grey
    6: "\033[45m",   # magenta
    7: "\033[48;5;202m",  # orange
    8: "\033[46m",   # cyan (azure)
    9: "\033[48;5;52m",   # dark red (maroon)
}
_RESET = "\033[0m"


class GridVisualizer:
    """Visualize ARC grids in terminal or matplotlib."""

    def print_grid(self, grid: ArcGrid, title: str = "") -> None:
        """Print a coloured grid to the terminal.

        Args:
            grid:  The ArcGrid to display.
            title: Optional header above the grid.
        """
        if title:
            print(f"\n{title}")
        print(f"({grid.height}×{grid.width})")
        for row in grid.pixels.tolist():
            row_str = ""
            for cell in row:
                color_code = _ANSI_COLORS.get(cell, "")
                row_str += f"{color_code}  {_RESET}"
            print(row_str)

    def print_pair(
        self,
        input_grid: ArcGrid,
        output_grid: ArcGrid,
        prediction: Optional[ArcGrid] = None,
    ) -> None:
        """Print input, output, and optionally prediction side-by-side."""
        self.print_grid(input_grid, "INPUT")
        self.print_grid(output_grid, "OUTPUT")
        if prediction is not None:
            match = "✓" if prediction == output_grid else "✗"
            self.print_grid(prediction, f"PREDICTION {match}")

    def print_task_summary(self, task: dict) -> None:
        """Print a summary of training pairs."""
        pairs = task.get("train", [])
        print(f"\nTask: {len(pairs)} training pairs")
        for i, pair in enumerate(pairs):
            print(f"\n--- Pair {i} ---")
            self.print_pair(pair["input"], pair["output"])

    def to_matplotlib(
        self,
        grids: List[ArcGrid],
        titles: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Render grids as a matplotlib figure.

        Args:
            grids:     List of ArcGrid objects to display.
            titles:    Optional list of subplot titles.
            save_path: If set, save the figure to this path.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            import numpy as np
        except ImportError:
            print("matplotlib not installed. Run: pip install matplotlib")
            return

        # ARC colour palette (RGB)
        palette = [
            [0, 0, 0],        # 0 black
            [0, 0, 255],      # 1 blue
            [255, 0, 0],      # 2 red
            [0, 128, 0],      # 3 green
            [255, 255, 0],    # 4 yellow
            [128, 128, 128],  # 5 grey
            [255, 0, 255],    # 6 magenta
            [255, 165, 0],    # 7 orange
            [0, 255, 255],    # 8 azure
            [128, 0, 0],      # 9 maroon
        ]
        cmap = mcolors.ListedColormap(
            [[r / 255, g / 255, b / 255] for r, g, b in palette]
        )

        n = len(grids)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, grid, title in zip(axes, grids, (titles or [""] * n)):
            ax.imshow(grid.pixels, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
            ax.set_title(title or "")
            ax.set_xticks([])
            ax.set_yticks([])
            # Grid lines
            ax.set_xticks([x - 0.5 for x in range(grid.width + 1)], minor=True)
            ax.set_yticks([y - 0.5 for y in range(grid.height + 1)], minor=True)
            ax.grid(which="minor", color="white", linewidth=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        else:
            plt.show()


__all__ = ["GridVisualizer"]
