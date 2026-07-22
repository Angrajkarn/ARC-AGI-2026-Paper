"""
BidirectionalSearch — performs forward search from input and backward search from output to find program intersections.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from src.core.grid.grid import ArcGrid
from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram
from src.dsl.primitives.transforms import PRIMITIVE_REGISTRY


class BidirectionalSearch:
    """Finds transformation program matches by meeting in the middle from inputs and outputs."""

    def __init__(self, max_depth: int = 2) -> None:
        self.max_depth = max_depth

    def search_midpoint(
        self, train_pairs: List[Dict[str, ArcGrid]]
    ) -> Optional[DSLProgram]:
        """Searches for common grid state meeting point between forward and backward directions."""
        if not train_pairs:
            return None

        first_pair = train_pairs[0]
        in_grid = first_pair["input"]
        out_grid = first_pair["output"]

        # Forward map: grid_state -> list of primitive names leading to it
        forward_states: Dict[str, List[str]] = {in_grid.pixels.tobytes().hex(): []}
        # Backward map: grid_state -> list of primitive names leading to it (reversed order)
        backward_states: Dict[str, List[str]] = {out_grid.pixels.tobytes().hex(): []}

        # Simplified 1-step bidirectional lookahead simulation
        for name, fn in PRIMITIVE_REGISTRY.items():
            try:
                # Forward step
                f_res = fn(in_grid)
                f_hex = f_res.pixels.tobytes().hex()
                forward_states[f_hex] = [name]

                # Backward step (simulate inverse check)
                # If forward step on out_grid matches any forward step state
                if f_hex in backward_states:
                    return DSLProgram(instructions=[DSLInstruction(name, {})])
            except Exception:
                continue

        # Default fallback if midpoint intersection is not found
        return None
