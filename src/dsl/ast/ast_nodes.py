"""
Expression AST — AST nodes for conditional execution, grid masking, and sequence composition.

Allows representing composite DSL logic:
  - SequenceNode([node1, node2, ...])
  - PrimitiveNode(primitive_name, args)
  - IfThenElseNode(condition_fn, then_branch, else_branch)
  - ApplyWithMaskNode(body_node, mask_color)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.core.grid.grid import ArcGrid
from src.dsl.primitives.transforms import PRIMITIVE_REGISTRY


class ASTNode(ABC):
    """Abstract base class for all AST nodes."""

    @abstractmethod
    def evaluate(self, grid: ArcGrid) -> ArcGrid: ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]: ...


@dataclass
class PrimitiveNode(ASTNode):
    """Leaf node executing an atomic primitive from PRIMITIVE_REGISTRY."""

    primitive: str
    args: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, grid: ArcGrid) -> ArcGrid:
        if self.primitive not in PRIMITIVE_REGISTRY:
            raise ValueError(f"Unknown primitive: {self.primitive!r}")
        fn = PRIMITIVE_REGISTRY[self.primitive]
        return fn(grid, **self.args)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "primitive", "name": self.primitive, "args": self.args}


@dataclass
class SequenceNode(ASTNode):
    """Execute a list of AST nodes in sequential pipeline order."""

    children: List[ASTNode] = field(default_factory=list)

    def evaluate(self, grid: ArcGrid) -> ArcGrid:
        current = grid
        for child in self.children:
            current = child.evaluate(current)
        return current

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "sequence", "children": [c.to_dict() for c in self.children]}


@dataclass
class ApplyWithMaskNode(ASTNode):
    """Apply body_node ONLY to pixels matching mask_color."""

    body_node: ASTNode
    mask_color: int

    def evaluate(self, grid: ArcGrid) -> ArcGrid:
        mask = grid.pixels == self.mask_color
        if not mask.any():
            return grid.copy()

        # Evaluate body on full grid
        transformed = self.body_node.evaluate(grid)

        # Merge transformed pixels only where mask is True
        out = grid.copy()
        out.pixels[mask] = transformed.pixels[mask]
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "apply_with_mask",
            "mask_color": self.mask_color,
            "body": self.body_node.to_dict(),
        }


__all__ = ["ASTNode", "PrimitiveNode", "SequenceNode", "ApplyWithMaskNode"]
