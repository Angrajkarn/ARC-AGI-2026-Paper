"""
Automated Research Paper Figure & LaTeX TikZ Generator — exports SVG and LaTeX TikZ grid figures.

Usage:
    python scripts/generate_paper_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
from rich.console import Console

from src.core.grid.grid import ArcGrid

console = Console()


def grid_to_svg(grid: ArcGrid, cell_size: int = 30) -> str:
    """Exports ArcGrid to standalone SVG string for publication figures."""
    h, w = grid.height, grid.width
    color_hex = {
        0: "#000000", 1: "#0074D9", 2: "#FF4136", 3: "#2ECC40", 4: "#FFDC00",
        5: "#AAAAAA", 6: "#F012BE", 7: "#FF851B", 8: "#7FDBFF", 9: "#870C25",
    }

    svg_parts = [
        f'<svg width="{w * cell_size}" height="{h * cell_size}" xmlns="http://www.w3.org/2000/svg">'
    ]
    for r in range(h):
        for c in range(w):
            bg = color_hex.get(grid.get(r, c), "#FFFFFF")
            x, y = c * cell_size, r * cell_size
            svg_parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{bg}" stroke="#333" stroke-width="1"/>'
            )
    svg_parts.append('</svg>')
    return "".join(svg_parts)


def grid_to_tikz(grid: ArcGrid) -> str:
    """Exports ArcGrid to LaTeX TikZ matrix specification."""
    h, w = grid.height, grid.width
    tikz_lines = [r"\begin{tikzpicture}[scale=0.5]"]
    for r in range(h):
        for c in range(w):
            val = grid.get(r, c)
            y = h - 1 - r
            tikz_lines.append(
                f"  \\filldraw[fill=color{val}, draw=black] ({c},{y}) rectangle ({c+1},{y+1});"
            )
    tikz_lines.append(r"\end{tikzpicture}")
    return "\n".join(tikz_lines)


def generate_paper_figures() -> bool:
    """Generates figure files in docs/figures."""
    console.print("[bold green]Generating publication figures and LaTeX TikZ specifications...[/bold green]\n")

    fig_dir = root_dir / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    g = ArcGrid(pixels=np.array([[1, 2], [3, 4]]), background=0)
    svg_content = grid_to_svg(g)
    tikz_content = grid_to_tikz(g)

    (fig_dir / "sample_grid.svg").write_text(svg_content, encoding="utf-8")
    (fig_dir / "sample_grid.tex").write_text(tikz_content, encoding="utf-8")

    console.print(f"[bold green]✔ Saved SVG to {fig_dir / 'sample_grid.svg'}[/bold green]")
    console.print(f"[bold green]✔ Saved TikZ code to {fig_dir / 'sample_grid.tex'}[/bold green]")
    return True


if __name__ == "__main__":
    generate_paper_figures()
