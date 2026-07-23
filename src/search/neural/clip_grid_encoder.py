"""
ClipGridEncoder — Aligns natural language DSL instruction strings with grid transformations.
"""

from __future__ import annotations

from typing import List

import numpy as np


class ClipGridEncoder:
    """Simulates a contrastive alignment encoder matching descriptions with grid feature shifts."""

    def __init__(self) -> None:
        # Predefined mapping of keyword categories to target transformation operations
        self.vocabulary = {
            "rotate": ["rotate_90", "rotate_180", "rotate_270"],
            "mirror": ["mirror_horizontal", "mirror_vertical"],
            "scale": ["scale_2x", "scale_3x"],
            "color": ["replace_color", "swap_colors"],
        }

    def compute_similarity(self, description: str, candidate_program_name: str) -> float:
        """Computes matching probability score between instruction text and program names."""
        desc_lower = description.lower()
        prog_lower = candidate_program_name.lower()

        matched_category = False
        score = 0.0

        for keyword, ops in self.vocabulary.items():
            if keyword in desc_lower:
                matched_category = True
                if any(op in prog_lower for op in ops):
                    score += 0.8
                else:
                    score -= 0.3

        # Direct token match bonus
        tokens = desc_lower.split()
        for t in tokens:
            if t in prog_lower:
                score += 0.2

        return float(np.clip(score, 0.0, 1.0))
