"""Checks on the gradient-boosted tree model, the last of the three model types."""

import pytest

from diamonds import data, features, metrics, models, split


@pytest.fixture(scope="module")
def parts():
    return split.split(features.add_log_columns(data.load()))


@pytest.fixture(scope="module")
def gbm(parts):
    train, _ = parts
    return models.fit_gbm(train)


def test_predicts_one_value_per_row(parts, gbm):
    _, test = parts
    assert gbm.predict(test).shape == (len(test),)


def test_importances_cover_every_feature(gbm):
    importances = gbm.importances()
    assert set(importances.index) == {features.LOG_CARAT, "depth", "table", *features.GRADES}


def test_carat_is_the_most_important_feature(gbm):
    """Matches the EDA and the linear model: size dominates every other feature."""
    assert gbm.importances().idxmax() == features.LOG_CARAT


def test_clears_a_loose_accuracy_floor(parts, gbm):
    _, test = parts
    scores = metrics.score(features.target(test), gbm.predict(test))
    assert scores["r2_log"] > 0.95
    assert scores["rmse"] < 1500


def test_beats_the_linear_baseline(parts, gbm):
    """The point of trees here: they should out-predict OLS, not just match it."""
    train, test = parts
    ols_rmse = metrics.score(features.target(test), models.fit_linear(train).predict(test))["rmse"]
    gbm_rmse = metrics.score(features.target(test), gbm.predict(test))["rmse"]
    assert gbm_rmse < ols_rmse


@pytest.mark.parametrize("size", list(features.SIZE_FEATURES))
@pytest.mark.parametrize("encoding", features.ENCODINGS)
def test_fits_every_feature_variant(parts, size, encoding):
    train, test = parts
    fitted = models.fit_gbm(train, size=size, encoding=encoding)
    scores = metrics.score(features.target(test), fitted.predict(test))
    assert scores["r2_log"] > 0.95
