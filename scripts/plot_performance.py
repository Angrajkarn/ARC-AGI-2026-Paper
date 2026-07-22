"""
Command Line Plotting & Statistics Suite — renders ASCII bar charts of solver results in console.

Usage:
    python scripts/plot_performance.py
"""

from __future__ import annotations

import sys
from typing import Dict

from rich.console import Console
from rich.table import Table

console = Console()


def render_ascii_bar(val: float, max_val: float = 100.0, width: int = 20) -> str:
    """Returns horizontal ASCII bar string representing progress/percentage."""
    if max_val <= 0:
        return ""
    filled = int((val / max_val) * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def plot_statistics(results: Dict[str, float]) -> None:
    """Renders statistical table with ASCII performance bars in console."""
    table = Table(title="ARC Solver Algorithm Benchmarks")
    table.add_column("Search Engine", style="cyan")
    table.add_column("Accuracy (%)", style="magenta")
    table.add_column("Performance Bar", style="green")

    for algo, acc in results.items():
        bar = render_ascii_bar(acc)
        table.add_row(algo.upper(), f"{acc:.1f}%", bar)

    console.print(table)


if __name__ == "__main__":
    dummy_data = {"beam": 12.5, "mcts": 8.0, "genetic": 15.0, "constraint": 5.0}
    plot_statistics(dummy_data)
