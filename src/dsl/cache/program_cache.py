"""
SemanticProgramCache — Output grid state hashing and deduplication for search algorithms.
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

from src.core.grid.grid import ArcGrid
from src.dsl.parser.dsl_parser import DSLProgram


class SemanticProgramCache:
    """Caches output grid states to prune semantically redundant program search paths."""

    def __init__(self) -> None:
        # Maps input_grid_hash -> set of output_grid_hashes
        self._seen_states: Dict[str, Set[str]] = {}

    def _hash_grid(self, grid: ArcGrid) -> str:
        return f"{grid.pixels.tobytes().hex()}_{grid.background}"

    def is_seen(self, input_grid: ArcGrid, output_grid: ArcGrid) -> bool:
        """Returns True if the output_grid state has already been evaluated for input_grid."""
        in_key = self._hash_grid(input_grid)
        out_key = self._hash_grid(output_grid)

        if in_key in self._seen_states and out_key in self._seen_states[in_key]:
            return True
        return False

    def add(self, input_grid: ArcGrid, output_grid: ArcGrid) -> None:
        """Records the (input_grid, output_grid) state transition in cache."""
        in_key = self._hash_grid(input_grid)
        out_key = self._hash_grid(output_grid)

        if in_key not in self._seen_states:
            self._seen_states[in_key] = set()

        self._seen_states[in_key].add(out_key)

    def clear(self) -> None:
        """Clears the cache."""
        self._seen_states.clear()
