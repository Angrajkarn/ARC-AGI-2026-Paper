"""
Unit tests for NodePruner.
"""

from __future__ import annotations

import pytest

from src.meta_learning.node_pruner import NodePruner


class TestNodePruner:
    def test_prune_redundancy_cancel(self):
        ops = ["crop", "mirror_horizontal", "mirror_horizontal", "scale"]
        pruned = NodePruner.prune_redundancy(ops)

        # mirror horizontal cancels out -> ["crop", "scale"]
        assert pruned == ["crop", "scale"]

    def test_prune_redundancy_keep(self):
        ops = ["crop", "rotate_90", "rotate_90", "scale"]
        pruned = NodePruner.prune_redundancy(ops)

        # rotate 90 + rotate 90 is NOT canceled (it's not rotate_180 or mirror)
        assert pruned == ["crop", "rotate_90", "rotate_90", "scale"]
