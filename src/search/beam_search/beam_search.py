"""
Beam Search — heuristic program search for ARC DSL synthesis.

Algorithm:
  1. Initialise beam from top hypotheses (seeded programs).
  2. At each step, expand each beam entry by:
     a. Appending each registered primitive.
     b. Modifying existing primitive arguments.
  3. Score expanded programs on training pairs.
  4. Prune beam to top-k by score.
  5. Stop when a perfect program is found or budget exhausted.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.core.grid.grid import ArcGrid
from src.dsl.executor.executor import DSLExecutor
from src.dsl.parser.dsl_parser import DSLInstruction, DSLProgram
from src.dsl.primitives.transforms import PRIMITIVE_REGISTRY
from src.reasoning.rule_discovery.rule_discoverer import Hypothesis
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BeamEntry:
    """A scored candidate in the beam."""

    program: DSLProgram
    score: float = 0.0
    train_accuracy: float = 0.0
    num_correct: int = 0
    depth: int = 0

    def __lt__(self, other: "BeamEntry") -> bool:
        return self.score < other.score


@dataclass
class BeamSearchResult:
    """Result returned by BeamSearch.search()."""

    best_program: Optional[DSLProgram]
    best_score: float
    all_candidates: List[BeamEntry]
    iterations: int
    elapsed_sec: float
    found_perfect: bool


class BeamSearch:
    """Beam search over the DSL program space.

    Usage::

        searcher = BeamSearch(beam_width=10, max_depth=8)
        result = searcher.search(hypotheses, pairs, time_budget=60)
    """

    # Primitives that can be seeded without arguments
    _NO_ARG_PRIMITIVES = [
        "mirror_horizontal", "mirror_vertical", "mirror_diagonal",
        "rotate_90", "rotate_180", "rotate_270",
        "crop_to_content", "fill_holes", "hollow",
        "normalize_colors",
    ]

    # Primitives with fixed argument templates to try
    _ARG_PRIMITIVES = [
        ("replace_color", {"source_color": s, "target_color": t})
        for s in range(10)
        for t in range(10)
        if s != t
    ] + [
        ("gravity", {"direction": d}) for d in ["down", "up", "left", "right"]
    ] + [
        ("add_border", {"color": c, "thickness": 1}) for c in range(10)
    ] + [
        ("outline", {"color": c}) for c in range(10)
    ] + [
        ("scale", {"factor": f}) for f in [2, 3]
    ] + [
        ("tile", {"rows": r, "cols": c}) for r in [1, 2, 3] for c in [1, 2, 3] if not (r == 1 and c == 1)
    ] + [
        ("swap_colors", {"color_a": a, "color_b": b})
        for a in range(10) for b in range(a + 1, 10)
    ]

    def __init__(
        self,
        beam_width: int = 10,
        max_depth: int = 8,
        score_weights: Optional[Dict[str, float]] = None,
        memoize: bool = True,
    ) -> None:
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.score_weights = score_weights or {
            "train_accuracy": 0.6,
            "simplicity": 0.2,
            "consistency": 0.2,
        }
        self.memoize = memoize
        self._executor = DSLExecutor(debug=False)
        self._memo: Dict[str, float] = {}

    def search(
        self,
        hypotheses: List[Hypothesis],
        pairs: List[Dict],
        time_budget: float = 60.0,
        max_iterations: int = 500,
    ) -> BeamSearchResult:
        """Run beam search.

        Args:
            hypotheses:    Initial ranked hypotheses (used as beam seeds).
            pairs:         Training pairs with "input"/"output" ArcGrid values.
            time_budget:   Wall-clock time limit in seconds.
            max_iterations: Maximum expansion iterations.

        Returns:
            BeamSearchResult with best program found.
        """
        t_start = time.perf_counter()
        iteration = 0
        best: Optional[BeamEntry] = None

        # Initialise beam from hypotheses + no-arg primitives
        beam = self._initialise_beam(hypotheses, pairs)

        # Check if any seed is already perfect
        for entry in beam:
            if entry.train_accuracy == 1.0:
                elapsed = time.perf_counter() - t_start
                logger.info(f"Beam: perfect seed found at init, score={entry.score:.3f}")
                return BeamSearchResult(
                    best_program=entry.program,
                    best_score=entry.score,
                    all_candidates=beam,
                    iterations=0,
                    elapsed_sec=elapsed,
                    found_perfect=True,
                )

        if beam:
            best = max(beam, key=lambda e: e.score)

        while iteration < max_iterations:
            elapsed = time.perf_counter() - t_start
            if elapsed > time_budget:
                logger.debug(f"Beam: time budget exhausted after {iteration} iterations")
                break

            # Expand
            new_entries: List[BeamEntry] = []
            for entry in beam:
                if entry.depth >= self.max_depth:
                    continue
                expansions = self._expand(entry, pairs)
                new_entries.extend(expansions)

                # Check for perfect solution
                for exp in expansions:
                    if exp.train_accuracy == 1.0:
                        elapsed = time.perf_counter() - t_start
                        logger.info(
                            f"Beam: perfect program found at iteration {iteration}, "
                            f"depth={exp.depth}, score={exp.score:.3f}"
                        )
                        return BeamSearchResult(
                            best_program=exp.program,
                            best_score=exp.score,
                            all_candidates=beam + new_entries,
                            iterations=iteration,
                            elapsed_sec=elapsed,
                            found_perfect=True,
                        )

            # Merge and prune
            all_entries = beam + new_entries
            all_entries.sort(key=lambda e: e.score, reverse=True)
            beam = all_entries[: self.beam_width]

            if beam:
                candidate = beam[0]
                if best is None or candidate.score > best.score:
                    best = candidate
                    logger.debug(
                        f"Beam iter {iteration}: best_score={best.score:.3f}, "
                        f"depth={best.depth}, acc={best.train_accuracy:.2f}"
                    )

            iteration += 1

        elapsed = time.perf_counter() - t_start
        return BeamSearchResult(
            best_program=best.program if best else None,
            best_score=best.score if best else 0.0,
            all_candidates=beam,
            iterations=iteration,
            elapsed_sec=elapsed,
            found_perfect=False,
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise_beam(
        self, hypotheses: List[Hypothesis], pairs: List[Dict]
    ) -> List[BeamEntry]:
        entries: List[BeamEntry] = []
        seen: Set[str] = set()

        # Seed from hypotheses
        for h in hypotheses:
            entry = self._score_entry(h.candidate_program, pairs, depth=h.candidate_program.length)
            key = repr(h.candidate_program)
            if key not in seen:
                seen.add(key)
                entries.append(entry)

        # Also seed with all single no-arg primitives
        for name in self._NO_ARG_PRIMITIVES:
            prog = DSLProgram(
                instructions=[DSLInstruction(primitive=name)],
                name=name,
            )
            key = repr(prog)
            if key not in seen:
                seen.add(key)
                entries.append(self._score_entry(prog, pairs, depth=1))

        entries.sort(key=lambda e: e.score, reverse=True)
        return entries[: self.beam_width]

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def _expand(self, entry: BeamEntry, pairs: List[Dict]) -> List[BeamEntry]:
        """Expand a beam entry by appending one new primitive instruction."""
        new_entries: List[BeamEntry] = []

        # Try all no-arg primitives
        for name in self._NO_ARG_PRIMITIVES:
            new_prog = self._append_instruction(
                entry.program, DSLInstruction(primitive=name)
            )
            score_entry = self._score_entry(new_prog, pairs, depth=entry.depth + 1)
            new_entries.append(score_entry)

        # Try arg-based primitives
        for name, args in self._ARG_PRIMITIVES:
            new_prog = self._append_instruction(
                entry.program, DSLInstruction(primitive=name, args=args)
            )
            score_entry = self._score_entry(new_prog, pairs, depth=entry.depth + 1)
            new_entries.append(score_entry)

        return new_entries

    def _append_instruction(self, program: DSLProgram, instr: DSLInstruction) -> DSLProgram:
        new_program = DSLProgram(
            instructions=list(program.instructions) + [instr],
            name=program.name,
            source="beam",
        )
        return new_program

    def _score_entry(
        self, program: DSLProgram, pairs: List[Dict], depth: int
    ) -> BeamEntry:
        prog_key = repr(program)
        if self.memoize and prog_key in self._memo:
            cached_acc, cached_heur = self._memo[prog_key]
            score = self._compute_score(cached_acc, cached_heur, depth)
            return BeamEntry(
                program=program,
                score=score,
                train_accuracy=cached_acc,
                num_correct=round(cached_acc * len(pairs)),
                depth=depth,
            )

        from src.search.heuristic_evaluator import HeuristicEvaluator
        evaluator = HeuristicEvaluator()

        correct = 0
        heuristic_scores = []
        for pair in pairs:
            result, _ = self._executor.execute(program, pair["input"])
            if result is not None:
                if result == pair["output"]:
                    correct += 1
                    heuristic_scores.append(1.0)
                else:
                    sim = evaluator.evaluate_similarity(result, pair["output"])
                    heuristic_scores.append(sim.total_score)
            else:
                heuristic_scores.append(0.0)

        accuracy = correct / max(len(pairs), 1)
        avg_heuristic = sum(heuristic_scores) / max(len(heuristic_scores), 1)

        if self.memoize:
            self._memo[prog_key] = (accuracy, avg_heuristic)

        score = self._compute_score(accuracy, avg_heuristic, depth)
        return BeamEntry(
            program=program,
            score=score,
            train_accuracy=accuracy,
            num_correct=correct,
            depth=depth,
        )

    def _compute_score(self, accuracy: float, avg_heuristic: float, depth: int) -> float:
        simplicity = max(0.0, 1.0 - depth / max(self.max_depth, 1))
        # If accuracy == 1.0, maximum score
        if accuracy == 1.0:
            return 1.0 + 0.1 * simplicity
        w = self.score_weights
        return (
            w.get("train_accuracy", 0.6) * accuracy
            + 0.3 * avg_heuristic
            + w.get("simplicity", 0.1) * simplicity
        )


__all__ = ["BeamSearch", "BeamSearchResult", "BeamEntry"]
