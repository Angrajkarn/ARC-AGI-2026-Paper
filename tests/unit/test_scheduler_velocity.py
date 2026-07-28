"""
Unit tests for SchedulerVelocityController.
"""

from __future__ import annotations

import pytest

from src.meta_learning.scheduler_velocity import SchedulerVelocityController


class TestSchedulerVelocityController:
    def test_compute_step_size_high_accuracy(self):
        # High accuracy -> increase step size
        step = SchedulerVelocityController.compute_step_size([0.9, 0.85, 0.95], base_step=1)
        assert step == 3

    def test_compute_step_size_low_accuracy(self):
        # Low accuracy -> decrease step size or keep min 1
        step = SchedulerVelocityController.compute_step_size([0.1, 0.2, 0.1], base_step=2)
        assert step == 1

    def test_compute_step_size_moderate_accuracy(self):
        step = SchedulerVelocityController.compute_step_size([0.5, 0.5, 0.5], base_step=1)
        assert step == 1
