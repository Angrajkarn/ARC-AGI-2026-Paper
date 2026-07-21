"""
Unit tests for verify_submission script.
"""

from __future__ import annotations

import pytest

from scripts.verify_submission import verify_kaggle_submission


class TestVerifySubmission:
    def test_verify_submission(self):
        success = verify_kaggle_submission()
        assert success is True
