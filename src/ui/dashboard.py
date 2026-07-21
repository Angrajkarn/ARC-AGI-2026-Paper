"""
ARCDashboard — Rich terminal UI layout for visualizing ARC grids, objects, and search progress.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

from src.core.grid.grid import ArcGrid
from src.core.objects.detector import ObjectDetector


class ARCDashboard:
    """Rich interactive dashboard layout for ARC task visual inspection."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()
        self.detector = ObjectDetector()

    def render_task_summary(self, task_id: str, train_pairs: List[Dict[str, ArcGrid]]) -> Panel:
        """Renders task summary panel."""
        num_pairs = len(train_pairs)
        first_input = train_pairs[0]["input"] if num_pairs > 0 else None
        dim_str = f"{first_input.height}x{first_input.width}" if first_input else "N/A"

        info_text = (
            f"[bold cyan]Task ID:[/bold cyan] {task_id}\n"
            f"[bold green]Train Pairs:[/bold green] {num_pairs}\n"
            f"[bold yellow]Sample Dimensions:[/bold yellow] {dim_str}"
        )
        return Panel(info_text, title="ARC Task Inspector", border_style="blue")

    def render_objects_table(self, grid: ArcGrid) -> Table:
        """Renders table of detected objects in grid."""
        objects = self.detector.detect(grid)
        table = Table(title=f"Detected Objects ({len(objects)})")
        table.add_column("Object ID", style="cyan")
        table.add_column("Color", style="magenta")
        table.add_column("Area", style="green")
        table.add_column("Bounding Box", style="yellow")

        for obj in objects:
            table.add_row(
                str(obj.object_id),
                str(obj.color),
                str(obj.area),
                f"{obj.bounding_box.height}x{obj.bounding_box.width}",
            )
        return table
