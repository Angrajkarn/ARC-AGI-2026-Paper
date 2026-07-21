"""
Streamlit Interactive Web Application — ARC Task Visualizer and Solver Inspector.

Usage:
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.grid.grid import ArcGrid
from src.core.objects.detector import ObjectDetector


class StreamlitARCApp:
    """Helper module for rendering ARC grids and object metadata in Streamlit UI."""

    def __init__(self) -> None:
        self.detector = ObjectDetector()

    def grid_to_html_table(self, grid: ArcGrid, cell_size: int = 25) -> str:
        """Converts ArcGrid pixels to an HTML table with official ARC color palette."""
        color_hex = {
            0: "#000000",  # Black
            1: "#0074D9",  # Blue
            2: "#FF4136",  # Red
            3: "#2ECC40",  # Green
            4: "#FFDC00",  # Yellow
            5: "#AAAAAA",  # Grey
            6: "#F012BE",  # Magenta
            7: "#FF851B",  # Orange
            8: "#7FDBFF",  # Teal
            9: "#870C25",  # Maroon
        }

        h, w = grid.height, grid.width
        rows_html = []
        for r in range(h):
            cols_html = []
            for c in range(w):
                val = grid.get(r, c)
                bg = color_hex.get(val, "#FFFFFF")
                cols_html.append(
                    f'<td style="background-color: {bg}; width: {cell_size}px; height: {cell_size}px; '
                    f'border: 1px solid #333; text-align: center;"></td>'
                )
            rows_html.append(f"<tr>{''.join(cols_html)}</tr>")

        return f'<table style="border-collapse: collapse;">{"".join(rows_html)}</table>'


def main() -> None:
    """Streamlit application entry point."""
    try:
        import streamlit as st

        st.set_page_config(page_title="ARC-AGI-2026 Visual Engine", layout="wide")
        st.title("🧩 ARC-AGI-2026 Reasoning Engine Inspector")

        app_helper = StreamlitARCApp()
        st.sidebar.header("Task Selector")

        task_id = st.sidebar.text_input("Task ID", value="007bbfb7")
        st.info(f"Inspecting Task: {task_id}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
