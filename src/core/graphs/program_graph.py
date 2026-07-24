"""
ProgramGraphEmbedder — Represents DSL program instructions as directed dependency graphs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.dsl.parser.dsl_parser import DSLProgram


class ProgramGraphEmbedder:
    """Encodes DSL programs into execution dependency graph representations."""

    @staticmethod
    def build_program_graph(program: DSLProgram) -> Dict[str, Any]:
        """Constructs adjacency list of operation step flows."""
        instructions = program.instructions
        nodes: List[str] = []
        edges: List[Tuple[int, int]] = []

        for idx, instr in enumerate(instructions):
            nodes.append(instr.primitive)
            # Add execution dependency edge to subsequent instruction steps
            if idx > 0:
                edges.append((idx - 1, idx))

        return {
            "nodes": nodes,
            "edges": edges,
        }
