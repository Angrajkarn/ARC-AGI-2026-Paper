"""
Candidate Ranker — score and rank candidate programs.

Scoring factors:
  - Training accuracy (exact match rate)
  - Program simplicity (shorter = simpler)
  - Consistency score (object-level)
  - Source algorithm confidence
  - LLM rank (if available)

Returns top-2 candidates per test input per ARC competition rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.dsl.parser.dsl_parser import DSLProgram
from src.verifier.verifier import VerificationResult
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ScoredCandidate:
    """A program with a composite score."""

    program: DSLProgram
    score: float
    train_accuracy: float
    simplicity: float
    object_consistency: float
    algorithm: str = "unknown"
    verification: Optional[VerificationResult] = None

    def __repr__(self) -> str:
        return (
            f"ScoredCandidate(score={self.score:.3f}, "
            f"acc={self.train_accuracy:.2f}, "
            f"simple={self.simplicity:.2f}, "
            f"algo={self.algorithm!r})"
        )


class CandidateRanker:
    """Score and rank candidate DSL programs.

    Usage::

        ranker = CandidateRanker()
        top2 = ranker.rank(candidates, verifier_results, n=2)
    """

    def __init__(
        self,
        weight_accuracy: float = 0.6,
        weight_simplicity: float = 0.2,
        weight_consistency: float = 0.2,
        max_program_length: int = 12,
    ) -> None:
        self.weight_accuracy = weight_accuracy
        self.weight_simplicity = weight_simplicity
        self.weight_consistency = weight_consistency
        self.max_program_length = max_program_length

    def rank(
        self,
        programs: List[DSLProgram],
        verifier_results: List[VerificationResult],
        algorithms: Optional[List[str]] = None,
        n: int = 2,
    ) -> List[ScoredCandidate]:
        """Rank programs and return top-n.

        Args:
            programs:         List of candidate DSLPrograms.
            verifier_results: Corresponding VerificationResults (same order).
            algorithms:       Algorithm that produced each program.
            n:                Number of candidates to return.

        Returns:
            Top-n ScoredCandidates sorted by score descending.
        """
        if not programs:
            return []

        algos = algorithms or ["unknown"] * len(programs)
        assert len(programs) == len(verifier_results) == len(algos)

        scored: List[ScoredCandidate] = []
        for prog, ver, algo in zip(programs, verifier_results, algos):
            scored.append(self._score(prog, ver, algo))

        scored.sort(key=lambda c: c.score, reverse=True)
        logger.debug(
            f"Ranker: {len(scored)} candidates, top score={scored[0].score:.3f} "
            f"if scored else N/A"
        )
        return scored[:n]

    def rank_from_search_results(
        self, search_results: List, pairs: List[Dict], n: int = 2
    ) -> List[ScoredCandidate]:
        """Convenience: rank directly from SearchResult objects.

        Args:
            search_results: List of SearchResult instances.
            pairs:          Training pairs for re-verification.
            n:              Number of top candidates.

        Returns:
            Top-n ScoredCandidates.
        """
        from src.verifier.verifier import Verifier

        verifier = Verifier()
        programs = []
        results = []
        algos = []

        for sr in search_results:
            if sr.best_program is not None:
                programs.append(sr.best_program)
                results.append(verifier.verify(sr.best_program, pairs))
                algos.append(sr.algorithm)

        return self.rank(programs, results, algos, n=n)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        program: DSLProgram,
        verification: VerificationResult,
        algorithm: str,
    ) -> ScoredCandidate:
        accuracy = verification.train_accuracy if hasattr(verification, "train_accuracy") else verification.score
        simplicity = max(0.0, 1.0 - program.length / max(self.max_program_length, 1))
        obj_cons = (
            sum(verification.object_scores) / max(len(verification.object_scores), 1)
            if verification.object_scores else accuracy
        )

        score = (
            self.weight_accuracy * accuracy
            + self.weight_simplicity * simplicity
            + self.weight_consistency * obj_cons
        )

        return ScoredCandidate(
            program=program,
            score=score,
            train_accuracy=accuracy,
            simplicity=simplicity,
            object_consistency=obj_cons,
            algorithm=algorithm,
            verification=verification,
        )


__all__ = ["CandidateRanker", "ScoredCandidate"]
