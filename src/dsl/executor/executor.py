"""
DSL Executor — compile and run DSLProgram instances on ArcGrid objects.

Features:
  - Deterministic execution with full debug trace
  - Condition evaluation at runtime
  - Timeout / step budget enforcement
  - Sandboxed: never modifies input grids in-place
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.grid.grid import ArcGrid
from src.core.objects.detector import ObjectDetector
from src.dsl.parser.dsl_parser import (
    Condition,
    ConditionType,
    DSLInstruction,
    DSLProgram,
)
from src.dsl.primitives.transforms import PRIMITIVE_REGISTRY
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Execution trace
# ---------------------------------------------------------------------------

@dataclass
class StepTrace:
    """Record of a single instruction execution."""

    step_index: int
    primitive: str
    args: Dict[str, Any]
    condition_result: bool
    skipped: bool
    success: bool
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    input_shape: tuple = ()
    output_shape: tuple = ()


@dataclass
class ExecutionTrace:
    """Complete trace of a program execution."""

    program_name: str
    steps: List[StepTrace] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    total_elapsed_ms: float = 0.0

    def summary(self) -> str:
        executed = sum(1 for s in self.steps if not s.skipped and s.success)
        skipped = sum(1 for s in self.steps if s.skipped)
        errors = sum(1 for s in self.steps if s.error)
        return (
            f"Program '{self.program_name}': "
            f"{executed} executed, {skipped} skipped, {errors} errors, "
            f"{self.total_elapsed_ms:.1f}ms total"
        )


# ---------------------------------------------------------------------------
# Condition evaluator
# ---------------------------------------------------------------------------

class ConditionEvaluator:
    """Evaluate Condition predicates against the current grid state."""

    def __init__(self, detector: Optional[ObjectDetector] = None) -> None:
        self._detector = detector or ObjectDetector()

    def evaluate(self, condition: Condition, grid: ArcGrid) -> bool:
        ct = condition.ctype
        p = condition.params

        if ct == ConditionType.ALWAYS:
            return True
        if ct == ConditionType.NEVER:
            return False

        if ct == ConditionType.COLOR_COUNT:
            color = p["color"]
            op = p["op"]        # "eq", "lt", "gt", "le", "ge"
            threshold = p["threshold"]
            count = int((grid.pixels == color).sum())
            return {
                "eq": count == threshold,
                "lt": count < threshold,
                "gt": count > threshold,
                "le": count <= threshold,
                "ge": count >= threshold,
            }[op]

        if ct == ConditionType.OBJECT_HAS_TAG:
            tag = p["tag"]
            objects = self._detector.detect(grid)
            return any(tag in obj.tags for obj in objects)

        if ct == ConditionType.GRID_SYMMETRIC:
            axis = p.get("axis", "horizontal")
            import numpy as np
            if axis == "horizontal":
                return bool(np.array_equal(grid.pixels, grid.pixels[:, ::-1]))
            return bool(np.array_equal(grid.pixels, grid.pixels[::-1, :]))

        logger.warning(f"Unknown condition type: {ct}")
        return True


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class DSLExecutor:
    """Execute DSLProgram instances on ArcGrid inputs.

    Usage::

        executor = DSLExecutor()
        result, trace = executor.execute(program, input_grid)
    """

    def __init__(
        self,
        max_steps: int = 100,
        timeout_sec: float = 5.0,
        debug: bool = False,
    ) -> None:
        """
        Args:
            max_steps:   Maximum instructions to execute (safety limit).
            timeout_sec: Wall-clock timeout in seconds.
            debug:       If True, log each step.
        """
        self.max_steps = max_steps
        self.timeout_sec = timeout_sec
        self.debug = debug
        self._cond_eval = ConditionEvaluator()

    def execute(
        self,
        program: DSLProgram,
        grid: ArcGrid,
    ) -> tuple[Optional[ArcGrid], ExecutionTrace]:
        """Execute a DSL program on a grid.

        Args:
            program: The DSLProgram to execute.
            grid:    Input ArcGrid (not mutated).

        Returns:
            Tuple of (result_grid_or_None, ExecutionTrace).
        """
        trace = ExecutionTrace(program_name=program.name or repr(program))
        current = grid.copy()
        t_start = time.perf_counter()

        try:
            for step_idx, instruction in enumerate(program.instructions):
                if step_idx >= self.max_steps:
                    trace.error = f"Exceeded max_steps={self.max_steps}"
                    trace.success = False
                    break

                elapsed = (time.perf_counter() - t_start) * 1000
                if elapsed / 1000 > self.timeout_sec:
                    trace.error = f"Timeout after {elapsed:.0f}ms"
                    trace.success = False
                    break

                step_trace = self._execute_step(step_idx, instruction, current)
                trace.steps.append(step_trace)

                if step_trace.success and not step_trace.skipped:
                    # The primitive may return a new grid — apply it
                    # (some primitives return lists, handled separately)
                    new_grid = self._call_primitive(instruction, current)
                    if isinstance(new_grid, ArcGrid):
                        current = new_grid

                elif not step_trace.success:
                    trace.success = False
                    trace.error = step_trace.error
                    break

        except Exception as e:
            trace.success = False
            trace.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"Executor crashed: {trace.error}")
            return None, trace

        trace.total_elapsed_ms = (time.perf_counter() - t_start) * 1000
        if self.debug:
            logger.debug(trace.summary())

        return current if trace.success else None, trace

    def execute_on_pairs(
        self,
        program: DSLProgram,
        pairs: List[Dict],
    ) -> List[tuple[Optional[ArcGrid], ExecutionTrace]]:
        """Execute a program on multiple (input, expected_output) pairs.

        Args:
            program: The DSLProgram.
            pairs:   List of dicts with "input" key (ArcGrid).

        Returns:
            List of (result, trace) per pair.
        """
        return [self.execute(program, pair["input"]) for pair in pairs]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _execute_step(
        self,
        step_idx: int,
        instruction: DSLInstruction,
        grid: ArcGrid,
    ) -> StepTrace:
        """Execute a single instruction step."""
        t0 = time.perf_counter()

        # Evaluate condition
        try:
            cond_result = self._cond_eval.evaluate(instruction.condition, grid)
        except Exception as e:
            return StepTrace(
                step_index=step_idx,
                primitive=instruction.primitive,
                args=instruction.args,
                condition_result=False,
                skipped=True,
                success=True,
                error=f"Condition error: {e}",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        if not cond_result:
            return StepTrace(
                step_index=step_idx,
                primitive=instruction.primitive,
                args=instruction.args,
                condition_result=False,
                skipped=True,
                success=True,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        # Execute primitive
        try:
            result = self._call_primitive(instruction, grid)
            success = True
            error = None
            out_shape = result.size if isinstance(result, ArcGrid) else ()
        except Exception as e:
            success = False
            error = f"{type(e).__name__}: {e}"
            out_shape = ()
            if self.debug:
                logger.debug(f"Step {step_idx} '{instruction.primitive}' failed: {error}")

        elapsed = (time.perf_counter() - t0) * 1000
        return StepTrace(
            step_index=step_idx,
            primitive=instruction.primitive,
            args=instruction.args,
            condition_result=cond_result,
            skipped=False,
            success=success,
            error=error,
            elapsed_ms=elapsed,
            input_shape=grid.size,
            output_shape=out_shape,
        )

    def _call_primitive(self, instruction: DSLInstruction, grid: ArcGrid) -> Any:
        """Look up and call the registered primitive."""
        fn = PRIMITIVE_REGISTRY.get(instruction.primitive)
        if fn is None:
            raise ValueError(f"Unknown primitive: '{instruction.primitive}'")
        return fn(grid, **instruction.args)


__all__ = ["DSLExecutor", "ExecutionTrace", "StepTrace"]
