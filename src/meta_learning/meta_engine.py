"""
MetaLearningEngine — Dynamically tunes solver configuration based on task complexity.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.core.grid.grid import ArcGrid


class MetaLearningEngine:
    """Configures reasoning parameters by analyzing task examples to prevent search space explosion."""

    @staticmethod
    def extract_task_features(pairs: List[Dict[str, ArcGrid]]) -> Dict[str, Any]:
        """Extracts statistical parameters from input-output grid pairs."""
        if not pairs:
            return {"max_size": 0, "unique_colors": 0, "size_changes": False}

        sizes = []
        colors = set()
        size_changes = False

        for pair in pairs:
            inp = pair["input"]
            out = pair.get("output")
            sizes.append(inp.height * inp.width)
            colors.update(np.unique(inp.pixels))

            if out is not None:
                sizes.append(out.height * out.width)
                colors.update(np.unique(out.pixels))
                if inp.height != out.height or inp.width != out.width:
                    size_changes = True

        return {
            "max_size": max(sizes),
            "unique_colors": len(colors),
            "size_changes": size_changes,
        }

    def recommend_config(self, pairs: List[Dict[str, ArcGrid]]) -> Dict[str, Any]:
        """Determines optimized beam width, search depth, and timeout budget based on features."""
        features = self.extract_task_features(pairs)
        max_size = features["max_size"]
        unique_colors = features["unique_colors"]
        size_changes = features["size_changes"]

        # Default configuration
        beam_width = 10
        max_depth = 6
        time_budget = 30.0

        # Adjust based on grid size complexity (conserve resource on massive grids)
        if max_size > 400:  # > 20x20
            beam_width = 4
            max_depth = 4
            time_budget = 15.0
        elif max_size > 100:  # > 10x10
            beam_width = 8
            max_depth = 5
            time_budget = 20.0

        # Dynamic search depth extension for scaling/resizing transformations
        if size_changes:
            max_depth += 1

        # Extend time budget for color-rich multi-color tasks
        if unique_colors > 6:
            time_budget += 10.0

        return {
            "beam_width": beam_width,
            "max_depth": max_depth,
            "time_budget": time_budget,
        }
