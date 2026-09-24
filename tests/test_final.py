"""Checks that the selected final model is the one PLAN.md picks, and that
it still holds up against the alternatives it was chosen over."""

import pytest

from diamonds import data, features, final, metrics, models, split


@pytest.fixture(scope="module")
def parts():
    return split.split(features.add_log_columns(data.load()))


@pytest.fixture(scope="module")
def fitted(parts):
    train, _ = parts
    return final.fit_final(train)


def test_final_model_uses_the_chosen_variant(fitted):
    assert fitted.size == final.SIZE
    assert fitted.encoding == final.ENCODING


@pytest.mark.parametrize(
    "competitor",
    [
        lambda train: models.fit_linear(train, size="carat", encoding="one-hot"),
        lambda train: models.fit_ridge(train, size="carat", encoding="one-hot"),
        lambda train: models.fit_lasso(train, size="carat", encoding="one-hot"),
    ],
)
def test_final_model_beats_every_alternative_it_was_compared_against(parts, fitted, competitor):
    """One-hot is the best encoding for the linear models (see PLAN.md) - the
    final model has to still win against each type's best variant, not just
    the ordinal-encoded ones used elsewhere in the test suite."""
    train, test = parts
    other_rmse = metrics.score(features.target(test), competitor(train).predict(test))["rmse"]
    final_rmse = metrics.score(features.target(test), fitted.predict(test))["rmse"]
    assert final_rmse < other_rmse


def test_final_model_clears_a_high_accuracy_floor(parts, fitted):
    """Tighter than the loose 0.95 floor used for individual model types
    elsewhere - this one is supposed to be the best, not just acceptable."""
    _, test = parts
    scores = metrics.score(features.target(test), fitted.predict(test))
    assert scores["r2_log"] > 0.98
    assert scores["rmse"] < 700
