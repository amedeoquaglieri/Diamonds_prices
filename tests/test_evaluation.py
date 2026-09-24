"""Checks that every model clears the evaluation plan's sanity checks:
error broken out by carat band, and no strong residual-vs-carat trend."""

import pytest

from diamonds import data, features, metrics, models, split

MAX_RESIDUAL_CARAT_CORRELATION = 0.2


@pytest.fixture(scope="module")
def parts():
    return split.split(features.add_log_columns(data.load()))


@pytest.fixture(scope="module")
def fitted(parts):
    train, _ = parts
    return {
        "ols": models.fit_linear(train),
        "ridge": models.fit_ridge(train),
        "lasso": models.fit_lasso(train),
        "gbm": models.fit_gbm(train),
    }


@pytest.mark.parametrize("name", ["ols", "ridge", "lasso", "gbm"])
def test_every_band_clears_a_loose_accuracy_floor(parts, fitted, name):
    """Bands are narrower than the whole dataset, so R2 is expected to be
    lower than the headline score (see metrics.score_by_band) - the floor
    here is looser than the overall-accuracy tests for the same reason."""
    _, test = parts
    model = fitted[name]
    by_band = metrics.score_by_band(test["carat_band"], features.target(test), model.predict(test))
    assert (by_band["r2_log"] > 0.5).all()


def test_bands_are_reported_smallest_to_largest(parts, fitted):
    _, test = parts
    by_band = metrics.score_by_band(
        test["carat_band"], features.target(test), fitted["ols"].predict(test)
    )
    assert list(by_band.index) == list(data.CARAT_BAND_LABELS)


def test_error_grows_with_carat_band(parts, fitted):
    """Larger stones cost more, so a fixed percentage error is a larger
    dollar error - RMSE should climb from the smallest band to the
    largest, not because the model fits big stones worse."""
    _, test = parts
    by_band = metrics.score_by_band(
        test["carat_band"], features.target(test), fitted["gbm"].predict(test)
    )
    assert by_band["rmse"].iloc[-1] > by_band["rmse"].iloc[0]


@pytest.mark.parametrize("name", ["ols", "ridge", "lasso", "gbm"])
def test_residual_has_no_strong_carat_trend(parts, fitted, name):
    """The confound this whole project is about is a carat trend hiding in
    a coefficient; this checks the mirror image, that none is left over in
    the error after fitting."""
    _, test = parts
    model = fitted[name]
    correlation = metrics.residual_carat_correlation(
        test["carat"], features.target(test), model.predict(test)
    )
    assert abs(correlation) < MAX_RESIDUAL_CARAT_CORRELATION
