"""Checks on partial dependence, the tree-model analogue of a linear coefficient."""

import pytest

from diamonds import data, features, interpret, models, split
from diamonds.data import GRADE_ORDERS


@pytest.fixture(scope="module")
def parts():
    return split.split(features.add_log_columns(data.load()))


@pytest.fixture(scope="module")
def gbm(parts):
    train, _ = parts
    return models.fit_gbm(train)


@pytest.mark.parametrize("grade", features.GRADES)
def test_partial_dependence_covers_every_grade_level(gbm, parts, grade):
    train, _ = parts
    assert list(interpret.partial_dependence(gbm, train, grade).index) == GRADE_ORDERS[grade]


@pytest.mark.parametrize("grade", features.GRADES)
def test_partial_dependence_increases_worst_to_best(gbm, parts, grade):
    """The confound check, restated for a model with no coefficients to read:
    predicted price must rise from worst to best grade once carat's effect
    is isolated, matching the OLS baseline's direction."""
    train, _ = parts
    values = list(interpret.partial_dependence(gbm, train, grade))
    assert values == sorted(values)


def test_partial_dependence_rejects_one_hot(parts):
    train, _ = parts
    one_hot_model = models.fit_gbm(train, encoding="one-hot")
    with pytest.raises(ValueError):
        interpret.partial_dependence(one_hot_model, train, "clarity")
