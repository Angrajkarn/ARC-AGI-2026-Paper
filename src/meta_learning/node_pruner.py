"""
NodePruner — Simplifies sequential DSL programs by filtering out redundant null operations.
"""

from __future__ import annotations

from typing import List


class NodePruner:
    """Parses program lists to remove operations that cancel out or produce identity states."""

    @staticmethod
    def prune_redundancy(ops: List[str]) -> List[str]:
        """Eliminates redundant adjacent pairs (e.g. double mirror or double 180 rotations)."""
        pruned = []
        i = 0
        n = len(ops)

        while i < n:
            if i < n - 1 and ops[i] == ops[i + 1] and ("rotate_180" in ops[i] or "mirror" in ops[i]):
                # Skip both since they cancel each other out
                i += 2
            else:
                pruned.append(ops[i])
                i += 1

        return pruned
