"""
Unit tests for build_kaggle_release script.
"""

from __future__ import annotations

import pytest

from scripts.build_kaggle_release import build_release


class TestReleaseBuilder:
    def test_build_release(self):
        success = build_release()
        assert success is True
