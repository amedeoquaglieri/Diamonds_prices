"""Checks on the OLS baseline, above all that it undoes the carat confound."""

import pytest

from diamonds import data, features, metrics, models, split
from diamonds.data import GRADE_ORDERS

# The EDA's unconditional log-log slope is 1.68. Controlling for the grades
# raises it, because better-quality stones skew smaller: the same confound,
# seen from the carat side. The band is wide on purpose.
MIN_CARAT_ELASTICITY = 1.5
MAX_CARAT_ELASTICITY = 2.2


@pytest.fixture(scope="module")
def parts():
    return split.split(features.add_log_columns(data.load()))


@pytest.fixture(scope="module")
def ordinal_model(parts):
    train, _ = parts
    return models.fit_linear(train, size="carat", encoding="ordinal")


@pytest.fixture(scope="module")
def one_hot_model(parts):
    train, _ = parts
    return models.fit_linear(train, size="carat", encoding="one-hot")


def test_fitted_keeps_its_feature_variant(ordinal_model):
    assert ordinal_model.size == "carat"
    assert ordinal_model.encoding == "ordinal"


def test_predicts_one_value_per_row(parts, ordinal_model):
    _, test = parts
    assert ordinal_model.predict(test).shape == (len(test),)


def test_carat_elasticity_is_near_the_eda_estimate(ordinal_model):
    elasticity = ordinal_model.coefficients()[features.LOG_CARAT]
    assert MIN_CARAT_ELASTICITY < elasticity < MAX_CARAT_ELASTICITY


def test_grade_coefficients_are_positive(ordinal_model):
    """The confound check: raw averages say better grades are cheaper, and
    the model must reverse that once carat is held fixed."""
    coefficients = ordinal_model.coefficients()
    for grade in features.GRADES:
        assert coefficients[grade] > 0


def test_one_hot_grades_increase_from_worst_to_best(one_hot_model):
    """One-hot encoding tells the model nothing about grade order, so a
    monotonic run of coefficients has to be learned from price alone."""
    coefficients = one_hot_model.coefficients()
    for grade, order in GRADE_ORDERS.items():
        values = [coefficients[f"{grade}_{level}"] for level in order[1:]]
        assert values == sorted(values)
        assert values[0] > 0


def test_baseline_clears_a_loose_accuracy_floor(parts, ordinal_model):
    """Guards against a refactor silently breaking training."""
    _, test = parts
    scores = metrics.score(features.target(test), ordinal_model.predict(test))
    assert scores["r2_log"] > 0.95
    assert scores["rmse"] < 1500


def test_model_does_not_overfit_badly(parts, ordinal_model):
    train, test = parts
    train_r2 = metrics.score(features.target(train), ordinal_model.predict(train))["r2_log"]
    test_r2 = metrics.score(features.target(test), ordinal_model.predict(test))["r2_log"]
    assert abs(train_r2 - test_r2) < 0.05
