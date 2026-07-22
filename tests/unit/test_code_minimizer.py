"""
Unit tests for SubmissionMinimizer.
"""

from __future__ import annotations

import pytest

from src.submission.code_minimizer import SubmissionMinimizer


class TestCodeMinimizer:
    def test_minimize_comments(self):
        source = (
            "# This is a comment\n"
            "def foo():\n"
            "    # Inline comment\n"
            "    return 42\n"
        )
        minimized = SubmissionMinimizer.minimize(source)
        assert "# This is a comment" not in minimized
        assert "# Inline comment" not in minimized
        assert "def foo():" in minimized
