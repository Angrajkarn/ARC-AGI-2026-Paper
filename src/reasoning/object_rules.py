"""
Object Rules Engine — infer object-level spatial movements and attribute transformations.

Induces rules such as:
  - Move object of color A to touch object of color B
  - Recolor objects based on area rank (largest -> color 1, 2nd largest -> color 2)
  - Select top-K objects by area/color to build the output grid
  - Align objects to canvas boundaries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.grid.grid import ArcGrid
from src.core.objects.arc_object import ArcObject
from src.core.objects.detector import ObjectDetector
from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ObjectRule:
    rule_type: str  # "area_rank_recolor", "isolate_object", "move_to_edge"
    description: str
    confidence: float
    program: DSLProgram
    params: Dict = field(default_factory=dict)


class ObjectRulesEngine:
    """Infer object-level spatial and attribute rules across training pairs."""

    def __init__(self) -> None:
        self._detector = ObjectDetector()

    def discover_object_rules(self, pairs: List[Dict]) -> List[ObjectRule]:
        """Discover object-level candidate rules across training pairs."""
        if not pairs:
            return []

        rules: List[ObjectRule] = []

        # Rule 1: Isolate largest object
        largest_rule = self._check_isolate_by_size(pairs, mode="largest")
        if largest_rule:
            rules.append(largest_rule)

        # Rule 2: Isolate smallest object
        smallest_rule = self._check_isolate_by_size(pairs, mode="smallest")
        if smallest_rule:
            rules.append(smallest_rule)

        # Rule 3: Isolate object of specific color
        color_rules = self._check_isolate_by_color(pairs)
        rules.extend(color_rules)

        rules.sort(key=lambda r: r.confidence, reverse=True)
        return rules

    def _check_isolate_by_size(
        self, pairs: List[Dict], mode: str = "largest"
    ) -> Optional[ObjectRule]:
        """Check if output grid equals isolating the largest/smallest object of input."""
        from src.dsl.executor.executor import DSLExecutor
        executor = DSLExecutor(debug=False)
        primitive_name = f"isolate_{mode}"
        prog = DSLProgram(instructions=[DSLInstruction(primitive=primitive_name)])

        correct = 0
        for p in pairs:
            res, _ = executor.execute(prog, p["input"])
            if res is not None and res == p["output"]:
                correct += 1

        conf = correct / max(len(pairs), 1)
        if conf > 0:
            return ObjectRule(
                rule_type="isolate_object",
                description=f"Isolate {mode} object",
                confidence=conf,
                program=prog,
                params={"mode": mode},
            )
        return None

    def _check_isolate_by_color(self, pairs: List[Dict]) -> List[ObjectRule]:
        """Check if output grid equals keeping only objects of a specific color."""
        from src.dsl.executor.executor import DSLExecutor
        executor = DSLExecutor(debug=False)
        results: List[ObjectRule] = []

        for color in range(10):
            prog = DSLProgram(
                instructions=[
                    DSLInstruction(
                        primitive="filter_objects_by_color", args={"color": color}
                    )
                ]
            )
            correct = 0
            for p in pairs:
                res, _ = executor.execute(prog, p["input"])
                if res is not None and res == p["output"]:
                    correct += 1
            conf = correct / max(len(pairs), 1)
            if conf > 0:
                results.append(
                    ObjectRule(
                        rule_type="isolate_by_color",
                        description=f"Filter objects by color {color}",
                        confidence=conf,
                        program=prog,
                        params={"color": color},
                    )
                )

        return results


__all__ = ["ObjectRulesEngine", "ObjectRule"]
