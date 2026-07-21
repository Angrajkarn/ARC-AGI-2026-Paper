"""
ASTDecompiler — Translates DSL programs into natural language explanations and LaTeX pseudo-code.
"""

from __future__ import annotations

from typing import List

from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram


class ASTDecompiler:
    """Decompiles DSL programs to human-readable explanations and LaTeX math formulations."""

    @staticmethod
    def to_natural_language(program: DSLProgram) -> str:
        """Translates DSLProgram to step-by-step natural language instructions."""
        if not program.instructions:
            return "1. Identity transformation (no changes)."

        steps = []
        for i, ins in enumerate(program.instructions, 1):
            name = ins.primitive
            args_str = ", ".join(f"{k}={v}" for k, v in ins.args.items())
            if args_str:
                steps.append(f"{i}. Apply primitive '{name}' with arguments ({args_str}).")
            else:
                steps.append(f"{i}. Apply primitive '{name}'.")

        return "\n".join(steps)

    @staticmethod
    def to_latex(program: DSLProgram) -> str:
        """Translates DSLProgram to LaTeX pseudo-code equation formulation."""
        if not program.instructions:
            return r"G_{\text{out}} = G_{\text{in}}"

        ops = []
        for ins in reversed(program.instructions):
            ops.append(r"\text{" + ins.primitive + r"}")

        comp_chain = r" \circ ".join(ops)
        return rf"G_{{\text{{out}}}} = \left( {comp_chain} \right)(G_{{\text{{in}}}})"
