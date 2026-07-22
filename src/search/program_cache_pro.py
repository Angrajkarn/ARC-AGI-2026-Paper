"""
SubtreeProgramCache — maps grid state pairs to synthesized DSL program subtrees.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.core.grid.grid import ArcGrid
from src.dsl.parser.dsl_parser import DSLProgram


class SubtreeProgramCache:
    """Manages cache lookup for intermediate grid transformation subtrees."""

    def __init__(self) -> None:
        self.cache: Dict[str, DSLProgram] = {}

    @staticmethod
    def _make_key(in_grid: ArcGrid, out_grid: ArcGrid) -> str:
        in_hex = in_grid.pixels.tobytes().hex()
        out_hex = out_grid.pixels.tobytes().hex()
        return f"{in_hex}:{out_hex}"

    def lookup(self, in_grid: ArcGrid, out_grid: ArcGrid) -> Optional[DSLProgram]:
        """Looks up cached sub-program matching input-output transition pair."""
        key = self._make_key(in_grid, out_grid)
        return self.cache.get(key)

    def insert(self, in_grid: ArcGrid, out_grid: ArcGrid, program: DSLProgram) -> None:
        """Inserts a verified sub-program mapping to the cache directory."""
        key = self._make_key(in_grid, out_grid)
        self.cache[key] = program
