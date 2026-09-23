"""Checks on the regularized regressions and how they compare to OLS."""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from diamonds import data, features, metrics, models, split


@pytest.fixture(scope="module")
def parts():
    return split.split(features.add_log_columns(data.load()))


@pytest.fixture(scope="module")
def ols(parts):
    train, _ = parts
    return models.fit_linear(train)


@pytest.fixture(scope="module")
def ridge(parts):
    train, _ = parts
    return models.fit_ridge(train)


@pytest.fixture(scope="module")
def lasso(parts):
    train, _ = parts
    return models.fit_lasso(train)


def test_ridge_coefficients_are_reported_in_original_units(ols, ridge):
    """Regularized models fit on standardized features; coefficients() must
    undo that, or they cannot be compared with the OLS baseline."""
    assert ridge.coefficients()[features.LOG_CARAT] == pytest.approx(
        ols.coefficients()[features.LOG_CARAT], rel=0.02
    )


def test_ridge_grade_coefficients_are_positive(ridge):
    coefficients = ridge.coefficients()
    for grade in features.GRADES:
        assert coefficients[grade] > 0


def test_ridge_barely_differs_from_ols_at_this_sample_size(parts, ols, ridge):
    """With ~43k rows and few features there is little variance to trade away,
    so ridge lands on essentially the OLS fit. Regularization is not what
    makes this problem work."""
    _, test = parts
    assert np.allclose(ols.predict(test), ridge.predict(test), atol=0.01)


def test_lasso_shrinks_total_coefficient_size(ols, lasso):
    assert lasso.coefficients().abs().sum() <= ols.coefficients().abs().sum()


@pytest.mark.parametrize("name", ["ridge", "lasso"])
def test_regularized_models_clear_a_loose_accuracy_floor(parts, name, request):
    _, test = parts
    model = request.getfixturevalue(name)
    scores = metrics.score(features.target(test), model.predict(test))
    assert scores["r2_log"] > 0.95
    assert scores["rmse"] < 1500


def test_ridge_alpha_comes_from_inside_the_search_grid(ridge):
    """An alpha at the edge of the grid would mean the grid is too narrow."""
    assert models.RIDGE_ALPHAS[0] < ridge.alpha < models.RIDGE_ALPHAS[-1]


def test_lasso_converges(parts):
    """Lasso on this data needs more iterations than the sklearn default."""
    train, _ = parts
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        models.fit_lasso(train, size="dimensions", encoding="one-hot")
