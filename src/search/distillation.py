"""
ProgramDistillation — Compiles frequently recurring DSL sub-sequences into unified macro-primitives.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from src.core.grid.grid import ArcGrid
from src.dsl.primitives.transforms import PRIMITIVE_REGISTRY


class ProgramDistillation:
    """Distills recurring sub-programs into new macro-primitives in the registry."""

    def __init__(self) -> None:
        self.registry = PRIMITIVE_REGISTRY

    def distill_sequence(self, sequence: List[str], macro_name: str) -> None:
        """Combines a list of existing primitive names into a unified callable macro-primitive."""
        if not sequence:
            return

        # Build list of callables corresponding to primitives
        callables = []
        for name in sequence:
            if name in self.registry:
                callables.append(self.registry[name])

        if not callables:
            return

        def macro_fn(grid: ArcGrid) -> ArcGrid:
            current = grid
            for fn in callables:
                current = fn(current)
            return current

        # Register the compiled macro-primitive dynamically
        self.registry[macro_name] = macro_fn
