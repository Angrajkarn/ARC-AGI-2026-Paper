"""
Unit tests for plot_performance script.
"""

from __future__ import annotations

import pytest

from scripts.plot_performance import plot_statistics, render_ascii_bar


class TestPlotPerformance:
    def test_render_ascii_bar(self):
        bar = render_ascii_bar(50.0, max_val=100.0, width=10)
        assert bar == "█████░░░░░"

    def test_plot_statistics(self):
        # Verify it runs without exceptions
        plot_statistics({"beam": 20.0, "mcts": 10.0})
