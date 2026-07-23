"""
ProgramReducer — Minimizes DSL program lengths by eliminating redundant operations.
"""

from __future__ import annotations

from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram


class ProgramReducer:
    """Optimizes synthesized programs to keep code clean and concise."""

    @staticmethod
    def reduce_program(program: DSLProgram) -> DSLProgram:
        """Removes consecutive mirror operations, identity rotations, and other redundancy loops."""
        instructions = program.instructions
        if not instructions:
            return program

        reduced: List[DSLInstruction] = []
        idx = 0
        while idx < len(instructions):
            curr = instructions[idx]

            # Check if we can collapse adjacent redundant mirror pairs
            if reduced and curr.primitive in ("mirror_horizontal", "mirror_vertical"):
                prev = reduced[-1]
                if prev.primitive == curr.primitive:
                    # Double mirror is an identity, delete previous and skip current
                    reduced.pop()
                    idx += 1
                    continue

            # Check if rotation collapses: e.g. rotate_180 + rotate_180 -> identity
            if reduced and curr.primitive == "rotate_180":
                prev = reduced[-1]
                if prev.primitive == "rotate_180":
                    reduced.pop()
                    idx += 1
                    continue

            reduced.append(curr)
            idx += 1

        return DSLProgram(
            instructions=reduced,
            name=program.name,
            source=program.source,
        )
