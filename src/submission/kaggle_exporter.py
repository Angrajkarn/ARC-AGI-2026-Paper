"""
Kaggle Notebook Bundle Exporter — packs the ARC reasoning engine into a self-contained notebook script.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


class KaggleExporter:
    """Generates a standalone Python submission script for Kaggle environment."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = root_dir or Path(__file__).resolve().parent.parent.parent

    def export(self, output_path: Path) -> Path:
        """Packs source code into output_path."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        header = [
            "# Auto-generated Kaggle Submission Script for ARC-AGI-2026 Engine\n",
            "# Environment: Offline Python 3.10+\n",
            "import os, sys, json, time\n",
            "import numpy as np\n\n",
        ]

        # Read core modules
        modules_to_pack = [
            "src/core/grid/grid.py",
            "src/core/objects/arc_object.py",
            "src/core/objects/detector.py",
            "src/dsl/primitives/transforms.py",
            "src/dsl/primitives/advanced_primitives.py",
            "src/dsl/primitives/higher_order.py",
            "src/dsl/executor/executor.py",
            "src/search/neural_heuristic.py",
            "src/submission/submission_generator.py",
        ]

        packed_content = list(header)

        for rel_path in modules_to_pack:
            abs_path = self.root_dir / rel_path
            if abs_path.exists():
                packed_content.append(f"# --- Begin {rel_path} ---\n")
                with open(abs_path, "r", encoding="utf-8") as f:
                    lines = [line for line in f if not line.startswith("from src.")]
                    packed_content.extend(lines)
                packed_content.append(f"\n# --- End {rel_path} ---\n\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(packed_content)

        return output_path
