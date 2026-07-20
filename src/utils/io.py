"""I/O utilities: ARC task loading, JSON helpers, path management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ARC Task Types
# ---------------------------------------------------------------------------

Grid = List[List[int]]  # 2D list of integers 0-9
Pair = Dict[str, Grid]  # {"input": Grid, "output": Grid}
ARCTask = Dict[str, Any]  # {"train": List[Pair], "test": List[Pair]}


def load_task(path: str | Path) -> ARCTask:
    """Load a single ARC task from a JSON file.

    Args:
        path: Path to the task JSON file.

    Returns:
        Parsed ARC task dict with "train" and "test" keys.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is not a valid ARC task.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        task = json.load(f)

    _validate_task(task)
    return task


def load_tasks_from_dir(
    directory: str | Path,
    max_tasks: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
) -> Dict[str, ARCTask]:
    """Load all ARC tasks from a directory.

    Args:
        directory: Directory containing .json task files.
        max_tasks: Limit number of tasks loaded (None = all).
        shuffle: Whether to shuffle the task list before selecting.
        seed: Random seed for shuffling.

    Returns:
        Dict mapping task_id (filename stem) to ARCTask.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    paths = sorted(directory.glob("*.json"))

    if shuffle:
        import random

        rng = random.Random(seed)
        rng.shuffle(paths)

    if max_tasks is not None:
        paths = paths[:max_tasks]

    tasks = {}
    for p in paths:
        try:
            tasks[p.stem] = load_task(p)
        except (ValueError, json.JSONDecodeError) as e:
            from src.utils.logging_utils import logger

            logger.warning(f"Skipping invalid task {p.name}: {e}")

    return tasks


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data as JSON to a file, creating parent directories as needed.

    Args:
        data: JSON-serializable data.
        path: Output file path.
        indent: JSON indentation (0 for compact).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent if indent > 0 else None)


def load_json(path: str | Path) -> Any:
    """Load JSON from a file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data.
    """
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_task(task: Any) -> None:
    """Validate that a parsed dict is a well-formed ARC task.

    Args:
        task: Parsed JSON object.

    Raises:
        ValueError: If the task structure is invalid.
    """
    if not isinstance(task, dict):
        raise ValueError("Task must be a JSON object")
    if "train" not in task or "test" not in task:
        raise ValueError("Task must have 'train' and 'test' keys")
    for split in ("train", "test"):
        if not isinstance(task[split], list):
            raise ValueError(f"'{split}' must be a list of pairs")
        for pair in task[split]:
            if "input" not in pair:
                raise ValueError(f"Each pair in '{split}' must have an 'input' key")
            _validate_grid(pair["input"], f"{split}.input")
            if "output" in pair:
                _validate_grid(pair["output"], f"{split}.output")


def _validate_grid(grid: Any, label: str) -> None:
    """Validate that a grid is a non-empty 2D list of ints 0-9.

    Args:
        grid: Object to validate.
        label: Human-readable label for error messages.

    Raises:
        ValueError: If the grid is malformed.
    """
    if not isinstance(grid, list) or len(grid) == 0:
        raise ValueError(f"{label} must be a non-empty list of rows")
    width = len(grid[0])
    for row in grid:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"{label} rows must all have the same length")
        for cell in row:
            if not isinstance(cell, int) or not (0 <= cell <= 9):
                raise ValueError(f"{label} cells must be integers 0-9, got {cell!r}")


def grid_to_tuple(grid: Grid) -> Tuple[Tuple[int, ...], ...]:
    """Convert a 2D list grid to a hashable tuple of tuples."""
    return tuple(tuple(row) for row in grid)


def tuple_to_grid(t: Tuple[Tuple[int, ...], ...]) -> Grid:
    """Convert a tuple of tuples back to a 2D list grid."""
    return [list(row) for row in t]


def ensure_dir(path: str | Path) -> Path:
    """Create directory and all parents if they don't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


__all__ = [
    "Grid",
    "Pair",
    "ARCTask",
    "load_task",
    "load_tasks_from_dir",
    "save_json",
    "load_json",
    "grid_to_tuple",
    "tuple_to_grid",
    "ensure_dir",
]
