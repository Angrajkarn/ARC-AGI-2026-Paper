"""
DSL Parser — parse and serialise ARC DSL programs.

A DSL program is a tree of DSLInstructions.  The textual representation is
a simple s-expression / JSON-like format that is human-readable and easy to
generate from LLMs or search algorithms.

Grammar (informal)::

    Program   := Instruction*
    Instruction := primitive_name [args] [IF condition]
    condition   := object_has_tag(tag) | color_count(color, cmp, n)
                 | always | never

Example JSON representation::

    [
      {"op": "mirror_horizontal"},
      {"op": "replace_color", "args": {"source_color": 2, "target_color": 5}},
      {"op": "rotate_90", "args": {"times": 1}, "condition": {"type": "always"}}
    ]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

from src.dsl.primitives.transforms import PRIMITIVE_REGISTRY


# ---------------------------------------------------------------------------
# Condition types
# ---------------------------------------------------------------------------

class ConditionType(Enum):
    ALWAYS = auto()
    NEVER = auto()
    COLOR_COUNT = auto()        # color_count(color, op, threshold)
    OBJECT_HAS_TAG = auto()     # object_has_tag(tag)
    GRID_SYMMETRIC = auto()     # grid is (h|v) symmetric


@dataclass
class Condition:
    """A predicate evaluated at runtime before executing an instruction."""

    ctype: ConditionType = ConditionType.ALWAYS
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def always(cls) -> "Condition":
        return cls(ctype=ConditionType.ALWAYS)

    @classmethod
    def never(cls) -> "Condition":
        return cls(ctype=ConditionType.NEVER)

    @classmethod
    def color_count(cls, color: int, op: str, threshold: int) -> "Condition":
        """Condition: count of *color* pixels satisfies op threshold."""
        return cls(ctype=ConditionType.COLOR_COUNT,
                   params={"color": color, "op": op, "threshold": threshold})

    @classmethod
    def object_has_tag(cls, tag: str) -> "Condition":
        return cls(ctype=ConditionType.OBJECT_HAS_TAG, params={"tag": tag})

    def to_dict(self) -> Dict:
        return {"type": self.ctype.name, "params": self.params}

    @classmethod
    def from_dict(cls, d: Dict) -> "Condition":
        ctype = ConditionType[d["type"]]
        return cls(ctype=ctype, params=d.get("params", {}))


# ---------------------------------------------------------------------------
# Instruction
# ---------------------------------------------------------------------------

@dataclass
class DSLInstruction:
    """A single step in a DSL program.

    Attributes:
        primitive:  Name of the primitive function (must be in PRIMITIVE_REGISTRY).
        args:       Keyword arguments forwarded to the primitive.
        condition:  Optional predicate; if False the instruction is skipped.
        comment:    Human-readable annotation.
    """

    primitive: str
    args: Dict[str, Any] = field(default_factory=dict)
    condition: Condition = field(default_factory=Condition.always)
    comment: str = ""

    def validate(self) -> None:
        """Raise ValueError if primitive is not registered."""
        if self.primitive not in PRIMITIVE_REGISTRY:
            raise ValueError(
                f"Unknown primitive '{self.primitive}'. "
                f"Available: {sorted(PRIMITIVE_REGISTRY.keys())}"
            )

    def to_dict(self) -> Dict:
        d: Dict = {"op": self.primitive}
        if self.args:
            d["args"] = self.args
        if self.condition.ctype != ConditionType.ALWAYS:
            d["condition"] = self.condition.to_dict()
        if self.comment:
            d["comment"] = self.comment
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "DSLInstruction":
        primitive = d["op"]
        args = d.get("args", {})
        cond_dict = d.get("condition", {"type": "ALWAYS", "params": {}})
        condition = Condition.from_dict(cond_dict)
        comment = d.get("comment", "")
        return cls(primitive=primitive, args=args, condition=condition, comment=comment)

    def __repr__(self) -> str:
        return f"DSLInstruction(op={self.primitive!r}, args={self.args})"


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------

@dataclass
class DSLProgram:
    """An ordered list of DSLInstructions forming a complete transformation.

    Attributes:
        instructions: Sequence of instructions to execute.
        name:         Optional human-readable name / description.
        source:       Where this program came from (beam, mcts, llm, etc.)
    """

    instructions: List[DSLInstruction] = field(default_factory=list)
    name: str = ""
    source: str = "unknown"

    @property
    def length(self) -> int:
        return len(self.instructions)

    def append(self, instruction: DSLInstruction) -> None:
        self.instructions.append(instruction)

    def validate(self) -> None:
        """Validate all instructions."""
        for i, instr in enumerate(self.instructions):
            try:
                instr.validate()
            except ValueError as e:
                raise ValueError(f"Instruction {i}: {e}") from e

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "source": self.source,
            "instructions": [i.to_dict() for i in self.instructions],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DSLProgram":
        instructions = [DSLInstruction.from_dict(i) for i in d.get("instructions", [])]
        return cls(
            instructions=instructions,
            name=d.get("name", ""),
            source=d.get("source", "unknown"),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "DSLProgram":
        return cls.from_dict(json.loads(s))

    def __repr__(self) -> str:
        ops = [f"{i.primitive}({i.args})" for i in self.instructions]
        return f"DSLProgram([{', '.join(ops)}])"

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        return iter(self.instructions)


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------

class DSLParser:
    """Parse DSL programs from various formats."""

    @staticmethod
    def from_json_string(s: str) -> DSLProgram:
        return DSLProgram.from_json(s)

    @staticmethod
    def from_op_list(ops: List[Dict[str, Any]]) -> DSLProgram:
        """Build a program from a plain list of op dicts.

        Example::
            [{"op": "rotate_90", "args": {"times": 1}}, {"op": "mirror_horizontal"}]
        """
        instructions = [DSLInstruction.from_dict(op) for op in ops]
        return DSLProgram(instructions=instructions)

    @staticmethod
    def from_names(names: List[str]) -> DSLProgram:
        """Build a program from a simple list of primitive names (no args)."""
        return DSLProgram(
            instructions=[DSLInstruction(primitive=n) for n in names]
        )

    @staticmethod
    def validate(program: DSLProgram) -> List[str]:
        """Return a list of validation error strings (empty if valid)."""
        errors = []
        for i, instr in enumerate(program.instructions):
            if instr.primitive not in PRIMITIVE_REGISTRY:
                errors.append(f"Step {i}: unknown primitive '{instr.primitive}'")
        return errors


__all__ = [
    "DSLProgram",
    "DSLInstruction",
    "Condition",
    "ConditionType",
    "DSLParser",
]
