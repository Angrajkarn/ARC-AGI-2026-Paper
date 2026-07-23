"""
ActiveCopilot — Interactively adapts search prioritize maps using human feedback/hints.
"""

from __future__ import annotations

from typing import Dict, List


class ActiveCopilot:
    """Active Learning assistant that takes user corrections to score candidate DSL nodes."""

    def __init__(self) -> None:
        self.user_hints: List[str] = []

    def register_feedback(self, hint: str) -> None:
        """Saves textual or component action correction hints."""
        if hint not in self.user_hints:
            self.user_hints.append(hint)

    def rerank_priors(self, candidate_names: List[str]) -> Dict[str, float]:
        """Calculates prioritized scores using registered guidance hints."""
        prioritized: Dict[str, float] = {}

        for cand in candidate_names:
            score = 1.0
            cand_lower = cand.lower()

            for hint in self.user_hints:
                hint_lower = hint.lower()
                # If candidate matches user's preferred action, boost priority
                if hint_lower in cand_lower:
                    score += 1.5
                # If user specifies action not wanted, penalize priority
                elif f"no_{hint_lower}" in cand_lower or f"avoid_{hint_lower}" in cand_lower:
                    score -= 0.8

            prioritized[cand] = max(0.1, score)

        return prioritized
