"""
Reflection Module — analyse failures and suggest improved search directions.

When a program fails verification, the Reflector:
  1. Identifies the nature of the mismatch (size, colour, object count, etc.)
  2. Categorises the failure type.
  3. Suggests targeted modifications.
  4. Generates new hypotheses or augmented programs.
  5. Logs a structured reasoning trace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from src.core.grid.grid import ArcGrid
from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram
from src.verifier.verifier import VerificationResult
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FailureType(Enum):
    EXECUTION_ERROR = auto()
    SIZE_MISMATCH = auto()
    COLOR_MISMATCH = auto()
    OBJECT_COUNT_MISMATCH = auto()
    PARTIAL_CORRECT = auto()
    ALL_WRONG = auto()
    UNKNOWN = auto()


@dataclass
class ReflectionReport:
    """Structured report of failure analysis and suggested actions."""

    failure_type: FailureType
    description: str
    affected_pairs: List[int] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    modified_programs: List[DSLProgram] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "failure_type": self.failure_type.name,
            "description": self.description,
            "affected_pairs": self.affected_pairs,
            "suggestions": self.suggestions,
            "confidence": self.confidence,
        }


class Reflector:
    """Analyse verification failures and propose improvements.

    Usage::

        reflector = Reflector()
        report = reflector.reflect(failed_program, pairs, verification_result)
        new_candidates = report.modified_programs
    """

    def reflect(
        self,
        program: DSLProgram,
        pairs: List[Dict],
        result: VerificationResult,
    ) -> ReflectionReport:
        """Analyse a failed verification result and produce a report.

        Args:
            program: The failing DSLProgram.
            pairs:   Training pairs.
            result:  VerificationResult from the Verifier.

        Returns:
            ReflectionReport with failure type, suggestions, and modified programs.
        """
        logger.debug(f"Reflecting on failure: {result}")

        failure_type = self._classify_failure(result, pairs)
        description = self._describe_failure(failure_type, result, pairs)
        suggestions = self._generate_suggestions(failure_type, result, pairs)
        modified = self._generate_modifications(program, failure_type, result, pairs)

        report = ReflectionReport(
            failure_type=failure_type,
            description=description,
            affected_pairs=self._affected_pairs(result),
            suggestions=suggestions,
            modified_programs=modified,
            confidence=result.score,
        )

        logger.debug(
            f"Reflection: {failure_type.name} — {len(modified)} modified programs suggested"
        )
        return report

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_failure(
        self, result: VerificationResult, pairs: List[Dict]
    ) -> FailureType:
        if result.num_correct == 0 and any(m == -1 for m in result.mismatch_pixels):
            return FailureType.EXECUTION_ERROR

        # Check size mismatches
        size_mismatches = [
            s for s in result.mismatch_sizes
            if s[1] is not None and s[0] != s[1]
        ]
        if size_mismatches:
            return FailureType.SIZE_MISMATCH

        if result.num_correct == 0:
            return FailureType.ALL_WRONG

        if 0 < result.num_correct < result.num_pairs:
            return FailureType.PARTIAL_CORRECT

        return FailureType.UNKNOWN

    def _describe_failure(
        self, failure_type: FailureType, result: VerificationResult, pairs: List[Dict]
    ) -> str:
        if failure_type == FailureType.EXECUTION_ERROR:
            return "Program raised an exception during execution."
        if failure_type == FailureType.SIZE_MISMATCH:
            sample = result.mismatch_sizes[0] if result.mismatch_sizes else ("?", "?")
            return f"Output size mismatch: expected {sample[0]}, got {sample[1]}."
        if failure_type == FailureType.ALL_WRONG:
            avg_mismatch = (
                sum(m for m in result.mismatch_pixels if m >= 0)
                / max(sum(1 for m in result.mismatch_pixels if m >= 0), 1)
            )
            return f"All pairs failed. Average pixel mismatch: {avg_mismatch:.0f} pixels."
        if failure_type == FailureType.PARTIAL_CORRECT:
            return (
                f"{result.num_correct}/{result.num_pairs} pairs correct. "
                f"Rule may not generalise."
            )
        return "Unknown failure mode."

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------

    def _generate_suggestions(
        self, failure_type: FailureType, result: VerificationResult, pairs: List[Dict]
    ) -> List[str]:
        suggestions = []

        if failure_type == FailureType.SIZE_MISMATCH:
            suggestions.append("Try crop_to_content to align output size.")
            suggestions.append("Check if the output is a sub-region of the input.")
            suggestions.append("Consider adding resize or scale operations.")

        if failure_type in (FailureType.ALL_WRONG, FailureType.PARTIAL_CORRECT):
            suggestions.append("Try reversing the order of operations.")
            suggestions.append("Consider adding a colour normalisation step.")
            suggestions.append("Try MCTS or genetic search for deeper exploration.")
            suggestions.append("Check whether the transformation is conditional on object properties.")

        if failure_type == FailureType.EXECUTION_ERROR:
            suggestions.append("Remove the last instruction and retry.")
            suggestions.append("Check argument types passed to primitives.")

        if failure_type == FailureType.PARTIAL_CORRECT:
            suggestions.append("The rule may need a conditional branch.")
            suggestions.append("Consider combining multiple primitives.")

        return suggestions

    # ------------------------------------------------------------------
    # Modified programs
    # ------------------------------------------------------------------

    def _generate_modifications(
        self,
        program: DSLProgram,
        failure_type: FailureType,
        result: VerificationResult,
        pairs: List[Dict],
    ) -> List[DSLProgram]:
        """Generate a list of modified programs to try next."""
        modifications: List[DSLProgram] = []

        instrs = list(program.instructions)

        if failure_type == FailureType.EXECUTION_ERROR and instrs:
            # Remove the last instruction
            modifications.append(
                DSLProgram(instructions=instrs[:-1], source="reflection")
            )

        if failure_type == FailureType.SIZE_MISMATCH:
            # Prepend crop_to_content
            modifications.append(
                DSLProgram(
                    instructions=[DSLInstruction(primitive="crop_to_content")] + instrs,
                    source="reflection",
                )
            )
            # Append crop_to_content
            modifications.append(
                DSLProgram(
                    instructions=instrs + [DSLInstruction(primitive="crop_to_content")],
                    source="reflection",
                )
            )

        if failure_type in (FailureType.ALL_WRONG, FailureType.PARTIAL_CORRECT):
            # Try reversed instruction order
            modifications.append(
                DSLProgram(instructions=list(reversed(instrs)), source="reflection")
            )
            # Try adding mirror operations
            for mirror_op in ["mirror_horizontal", "mirror_vertical"]:
                modifications.append(
                    DSLProgram(
                        instructions=[DSLInstruction(primitive=mirror_op)] + instrs,
                        source="reflection",
                    )
                )

        return modifications

    def _affected_pairs(self, result: VerificationResult) -> List[int]:
        """Return indices of pairs that failed."""
        per_pair = result.details.get("per_pair", [])
        return [d["pair"] for d in per_pair if not d.get("exact_match", False)]


__all__ = ["Reflector", "ReflectionReport", "FailureType"]
