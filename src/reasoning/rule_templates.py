"""
Rule Template Synthesizer — generate multi-step composite DSL program hypotheses.

Synthesizes 2-step and 3-step candidate programs:
  - crop_to_content -> rotate / mirror / scale
  - isolate_largest -> hollow / add_border / recolor_non_background
  - filter_objects_by_color -> replace_color
  - connect_points_of_color -> fill_holes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.core.grid.grid import ArcGrid
from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TemplateHypothesis:
    description: str
    program: DSLProgram
    confidence: float


class RuleTemplateSynthesizer:
    """Synthesizes 2-step and 3-step composite DSL program hypotheses."""

    def synthesize_templates(self, pairs: List[Dict]) -> List[TemplateHypothesis]:
        """Generate composite program candidates and score them against training pairs.

        Args:
            pairs: List of dicts with "input" and "output" ArcGrids.

        Returns:
            List of TemplateHypothesis instances sorted by confidence descending.
        """
        if not pairs:
            return []

        from src.dsl.executor.executor import DSLExecutor
        executor = DSLExecutor(debug=False)
        candidates: List[TemplateHypothesis] = []

        # Define 2-step and 3-step template patterns
        templates = [
            # Crop + Rotate / Mirror
            [{"op": "crop_to_content"}, {"op": "rotate_90", "args": {"times": 1}}],
            [{"op": "crop_to_content"}, {"op": "rotate_90", "args": {"times": 2}}],
            [{"op": "crop_to_content"}, {"op": "rotate_90", "args": {"times": 3}}],
            [{"op": "crop_to_content"}, {"op": "mirror_horizontal"}],
            [{"op": "crop_to_content"}, {"op": "mirror_vertical"}],
            [{"op": "crop_to_content"}, {"op": "scale", "args": {"factor": 2}}],

            # Isolate + Transform
            [{"op": "isolate_largest"}, {"op": "hollow"}],
            [{"op": "isolate_largest"}, {"op": "add_border", "args": {"color": 1, "thickness": 1}}],
            [{"op": "isolate_largest"}, {"op": "recolor_non_background", "args": {"new_color": 2}}],
            [{"op": "isolate_smallest"}, {"op": "scale", "args": {"factor": 2}}],

            # Drawing + Fill
            [{"op": "connect_points_of_color", "args": {"color": 1}}, {"op": "fill_holes"}],
            [{"op": "connect_points_of_color", "args": {"color": 2}}, {"op": "fill_holes"}],

            # Symmetry + Transform
            [{"op": "complete_symmetry_h"}, {"op": "fill_holes"}],
            [{"op": "complete_symmetry_v"}, {"op": "fill_holes"}],
        ]

        for template_ops in templates:
            instructions = [
                DSLInstruction(primitive=op["op"], args=op.get("args", {}))
                for op in template_ops
            ]
            op_names = " -> ".join(op["op"] for op in template_ops)
            prog = DSLProgram(instructions=instructions, source="template_synthesizer")

            correct = 0
            for p in pairs:
                res, trace = executor.execute(prog, p["input"])
                if res is not None and res == p["output"]:
                    correct += 1

            conf = correct / max(len(pairs), 1)
            if conf > 0:
                candidates.append(
                    TemplateHypothesis(
                        description=f"Composite template: {op_names}",
                        program=prog,
                        confidence=conf,
                    )
                )

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates


__all__ = ["RuleTemplateSynthesizer", "TemplateHypothesis"]
