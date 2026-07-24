"""
ProgramDistance — Computes Levenshtein edit distance between lists of primitive steps of two DSL programs.
"""

from __future__ import annotations

from typing import List


class ProgramDistance:
    """Estimates distance metrics between DSL program configurations."""

    @staticmethod
    def edit_distance(ops1: List[str], ops2: List[str]) -> int:
        """Standard dynamic programming Levenshtein distance."""
        m, n = len(ops1), len(ops2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ops1[i - 1] == ops2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Deletion
                        dp[i][j - 1],      # Insertion
                        dp[i - 1][j - 1]   # Substitution
                    )

        return dp[m][n]
