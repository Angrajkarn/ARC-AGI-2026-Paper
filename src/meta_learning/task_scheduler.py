"""
TaskCurriculumScheduler — Evaluates task complexity metrics to schedule task training ordering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class TaskCurriculumScheduler:
    """Sorts tasks by structural complexity parameters (e.g. dimensions, color counts)."""

    @staticmethod
    def get_complexity_score(task_file: Path) -> float:
        """Computes complexity score based on training grid size and colors count."""
        try:
            with open(task_file, "r", encoding="utf-8") as f:
                task_data = json.load(f)
            train_examples = task_data.get("train", [])
            if not train_examples:
                return float("inf")

            scores = []
            for ex in train_examples:
                inp = np.array(ex["input"])
                out = np.array(ex["output"])
                size_factor = float(inp.size + out.size)
                colors_factor = float(len(set(inp.flat) | set(out.flat)))
                scores.append(size_factor * colors_factor)
            return float(np.mean(scores))
        except Exception:
            return float("inf")

    def schedule_curriculum(self, task_dir: Path) -> List[Path]:
        """Returns list of JSON files sorted ascending by complexity score."""
        task_files = list(task_dir.glob("*.json"))
        # Sort based on complexity metric
        task_files.sort(key=self.get_complexity_score)
        return task_files
