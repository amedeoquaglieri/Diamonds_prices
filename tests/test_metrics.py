"""Checks on scoring, especially the log-to-dollars back-transformation."""

import numpy as np
import pytest

from diamonds import metrics


def test_perfect_prediction_scores_perfectly():
    log_true = np.log([500.0, 1500.0, 9000.0])
    scores = metrics.score(log_true, log_true)
    assert scores["rmse"] == 0.0
    assert scores["mae"] == 0.0
    assert scores["r2_log"] == 1.0


def test_errors_are_reported_in_dollars_not_log_units():
    """Hand-built case: predictions are $10 off on either side."""
    log_true = np.log([100.0, 200.0])
    log_pred = np.log([110.0, 190.0])
    scores = metrics.score(log_true, log_pred)
    assert scores["rmse"] == pytest.approx(10.0)
    assert scores["mae"] == pytest.approx(10.0)


def test_r2_is_measured_on_the_log_scale():
    """R2 must come from the log values, where the model is fitted."""
    log_true = np.log([100.0, 1000.0, 10000.0])
    log_pred = log_true + 0.1
    expected = 1 - (3 * 0.1**2) / np.sum((log_true - log_true.mean()) ** 2)
    assert np.isclose(metrics.score(log_true, log_pred)["r2_log"], expected)
