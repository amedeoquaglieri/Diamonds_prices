"""Checks on scoring, especially the log-to-dollars back-transformation."""

import numpy as np
import pandas as pd
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


def test_score_by_band_scores_each_band_on_its_own_rows():
    """Hand-built case: one band is a perfect fit, the other is $10 off."""
    carat_band = pd.Series(pd.Categorical(["small", "small", "big", "big"], categories=["small", "big"], ordered=True))
    log_true = np.log([100.0, 200.0, 500.0, 500.0])
    log_pred = np.log([100.0, 200.0, 510.0, 490.0])
    by_band = metrics.score_by_band(carat_band, log_true, log_pred)
    assert list(by_band.index) == ["small", "big"]
    assert by_band.loc["small", "rmse"] == pytest.approx(0.0)
    assert by_band.loc["big", "mae"] == pytest.approx(10.0)


def test_score_by_band_keeps_bands_in_category_order_not_alphabetical():
    """The category order is 'z' then 'a' - alphabetical would get this
    backwards, which is exactly the bug score_by_band has to avoid for the
    real 0.2-0.4, ..., 1.5+ band labels."""
    carat_band = pd.Series(
        pd.Categorical(["a", "a", "z", "z"], categories=["z", "a"], ordered=True)
    )
    log_true = log_pred = np.log([100.0, 200.0, 300.0, 400.0])
    by_band = metrics.score_by_band(carat_band, log_true, log_pred)
    assert list(by_band.index) == ["z", "a"]


def test_residual_carat_correlation_is_zero_with_no_size_trend():
    carat = pd.Series([0.3, 0.5, 1.0, 1.5, 2.0])
    log_true = np.log([300.0, 500.0, 1000.0, 1500.0, 2000.0])
    noise = np.log([310.0, 490.0, 1010.0, 1490.0, 2010.0])  # errors don't grow with carat
    assert abs(metrics.residual_carat_correlation(carat, log_true, noise)) < 0.5


def test_residual_carat_correlation_is_strongly_positive_with_a_size_trend():
    """Hand-built case: error grows with carat, so the residual should
    correlate strongly and positively with it."""
    carat = pd.Series([0.3, 0.5, 1.0, 1.5, 2.0])
    log_true = np.log([300.0, 500.0, 1000.0, 1500.0, 2000.0])
    log_pred = np.log([300.0, 490.0, 950.0, 1350.0, 1700.0])  # under-predicts more as carat grows
    assert metrics.residual_carat_correlation(carat, log_true, log_pred) == pytest.approx(1.0)
